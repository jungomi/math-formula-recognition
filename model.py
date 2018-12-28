import torch
import torch.nn as nn

# Number of bottlenecks
num_bn = 3
# The depth is half of the actual values in the paper because bottleneck blocks
# are used which contain two convlutional layers
depth = 16
multi_block_depth = depth // 2
growth_rate = 24

n = 256
n_prime = 512
decoder_conv_filters = 256
gru_hidden_size = 256
embedding_dim = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """Multi-scale Encoder

    A multi-scale encoder with two branches. The first branch produces
    low-resolution annotations, as a regular encoder would, and the second branch
    produces high-resolution annotations.
    """

    def __init__(self, img_channels=3, num_in_features=64, checkpoint=None):
        """
        Args:
            img_channels (int, optional): Number of channels of the images [Default: 3]
            num_in_features (int, optional): Number of channels that are created from
                the input to feed to the first dense block [Default: 64]
            checkpoint (dict, optional): State dictionary to be loaded
        """
        super(Encoder, self).__init__()
        self.conv0 = nn.Conv2d(
            img_channels,
            num_in_features,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.norm0 = nn.BatchNorm2d(num_in_features)
        self.conv1 = nn.Conv2d(num_in_features, 128, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(512)
        self.conv3_high = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.norm3_high = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if checkpoint is not None:
            self.load_state_dict(checkpoint)

    def forward(self, x):
        out = self.conv0(x)
        out = self.norm0(out)
        out = self.relu(out)
        out = self.max_pool(out)

        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.max_pool(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out_before_max = self.relu(out)
        out_low = self.max_pool(out_before_max)
        out_low = self.conv3(out_low)
        out_low = self.norm3(out_low)
        out_low = self.relu(out_low)
        out_high = self.conv3_high(out_before_max)
        out_high = self.norm3_high(out_high)
        out_high = self.relu(out_high)

        return out_low, out_high


class CoverageAttention(nn.Module):
    """Coverage attention

    The coverage attention is a multi-layer perceptron, which takes encoded annotations
    and creates a context vector.
    """

    # input_size = C
    # output_size = q
    # attn_size = L = H * W
    def __init__(
        self, input_size, output_size, attn_size, kernel_size, padding=0, device=device
    ):
        """
        Args:
            input_size (int): Number of channels of the input
            output_size (int): Number of channels of the coverage
            attn_size (int): Length of the annotation vector
            kernel_size (int): Kernel size of the 1D convolutional layer
            padding (int, optional): Padding of the 1D convolutional layer [Default: 0]
            device (torch.device, optional): Device for the tensors
        """
        super(CoverageAttention, self).__init__()
        self.alpha = None
        self.conv = nn.Conv2d(1, output_size, kernel_size=kernel_size, padding=padding)
        self.U_a = nn.Parameter(torch.empty((n_prime, input_size)))
        self.U_f = nn.Parameter(torch.empty((n_prime, output_size)))
        self.nu_attn = nn.Parameter(torch.empty(n_prime))
        self.input_size = input_size
        self.output_size = output_size
        self.attn_size = attn_size
        self.device = device
        nn.init.xavier_normal_(self.U_a)
        nn.init.xavier_normal_(self.U_f)
        # Xavier requires at least a 2D tensor.
        nn.init.xavier_normal_(self.nu_attn.unsqueeze(0))

    def reset_alpha(self, batch_size):
        self.alpha = torch.zeros((batch_size, 1, self.attn_size), device=self.device)

    def forward(self, x, u_pred):
        batch_size = x.size(0)
        if self.alpha is None:
            self.reset_alpha(batch_size)
        # Change the dimensions to make it possible to apply a 2D convolution
        # From: (batch_size x L)
        # To: (batch_size x H x W)
        alpha_sum = self.alpha.sum(1).view(batch_size, x.size(2), x.size(3))
        conv_out = self.conv(alpha_sum.unsqueeze(1))
        # Change dimensions back
        # From: (batch_size x output_size x H x W)
        # To: (batch_size x output_size x L)
        conv_out = conv_out.view(batch_size, self.output_size, -1)
        # Change the dimensions
        # From: (batch_size x C x H x W)
        # To: (batch_size x C x L)
        a = x.view(batch_size, x.size(1), -1)
        u_a = torch.matmul(self.U_a, a)
        u_f = torch.matmul(self.U_f, conv_out)
        # u_pred is expanded from (batch_size x n_prime)
        # to (batch_size x n_prime x L) because there are L components to which
        # the same u_pred is added.
        u_pred_expanded = u_pred.unsqueeze(2).expand_as(u_a)
        tan_res = torch.tanh(u_pred_expanded + u_a + u_f)
        e_t = torch.matmul(self.nu_attn, tan_res)
        alpha_t = torch.softmax(e_t, dim=1)
        self.alpha = torch.cat((self.alpha, alpha_t.detach().unsqueeze(1)), dim=1)
        # alpha_t: (batch_size x L)
        # a: (batch_size x C x L) but need (C x batch_size x L) for
        # element-wise multiplication. So transpose them.
        cA_t_L = alpha_t * a.transpose(0, 1)
        # Transpose back
        return cA_t_L.transpose(0, 1).sum(2)


class Maxout(nn.Module):
    """
    Maxout makes pools from the last dimension and keeps only the maximum value from
    each pool.
    """

    def __init__(self, pool_size):
        """
        Args:
            pool_size (int): Number of elements per pool
        """
        super(Maxout, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        [*shape, last] = x.size()
        out = x.view(*shape, last // self.pool_size, self.pool_size)
        out, _ = out.max(-1)
        return out


class Decoder(nn.Module):
    """Decoder

    GRU based Decoder which attends to the low- and high-resolution annotations to
    create a LaTeX string.
    """

    def __init__(
        self,
        num_classes,
        low_res_shape,
        high_res_shape,
        hidden_size=256,
        embedding_dim=256,
        checkpoint=None,
        device=device,
    ):
        """
        Args:
            num_classes (int): Number of symbol classes
            low_res_shape ((int, int, int)): Shape of the low resolution annotations
                i.e. (C, W, H)
            high_res_shape ((int, int, int)): Shape of the high resolution annotations
                i.e. (C_prime, 2W, 2H)
            hidden_size (int, optional): Hidden size of the GRU [Default: 256]
            embedding_dim (int, optional): Dimension of the embedding [Default: 256]
            checkpoint (dict, optional): State dictionary to be loaded
            device (torch.device, optional): Device for the tensors
        """
        super(Decoder, self).__init__()
        C = low_res_shape[0]
        C_prime = high_res_shape[0]
        context_size = C + C_prime
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.gru1 = nn.GRU(
            input_size=embedding_dim, hidden_size=hidden_size, batch_first=True
        )
        self.gru2 = nn.GRU(
            input_size=context_size, hidden_size=hidden_size, batch_first=True
        )
        # L = H * W
        low_res_attn_size = low_res_shape[1] * low_res_shape[2]
        high_res_attn_size = high_res_shape[1] * high_res_shape[2]
        self.coverage_attn_low = CoverageAttention(
            C,
            decoder_conv_filters,
            attn_size=low_res_attn_size,
            kernel_size=(11, 11),
            padding=5,
            device=device,
        )
        self.coverage_attn_high = CoverageAttention(
            C_prime,
            decoder_conv_filters,
            attn_size=high_res_attn_size,
            kernel_size=(7, 7),
            padding=3,
            device=device,
        )
        self.W_o = nn.Parameter(torch.empty((num_classes, embedding_dim // 2)))
        self.W_s = nn.Parameter(torch.empty((embedding_dim, hidden_size)))
        self.W_c = nn.Parameter(torch.empty((embedding_dim, context_size)))
        self.U_pred = nn.Parameter(torch.empty((n_prime, n)))
        self.maxout = Maxout(2)
        self.hidden_size = hidden_size
        nn.init.xavier_normal_(self.W_o)
        nn.init.xavier_normal_(self.W_s)
        nn.init.xavier_normal_(self.W_c)
        nn.init.xavier_normal_(self.U_pred)

        if checkpoint is not None:
            self.load_state_dict(checkpoint)

    def init_hidden(self, batch_size):
        return torch.zeros((1, batch_size, self.hidden_size))

    def reset(self, batch_size):
        self.coverage_attn_low.reset_alpha(batch_size)
        self.coverage_attn_high.reset_alpha(batch_size)

    # Unsqueeze and squeeze are used to add and remove the seq_len dimension,
    # which is always 1 since only the previous symbol is provided, not a sequence.
    # The inputs that are multiplied by the weights are transposed to get
    # (m x batch_size) instead of (batch_size x m). The result of the
    # multiplication is tranposed back.
    def forward(self, x, hidden, low_res, high_res):
        embedded = self.embedding(x)
        pred, _ = self.gru1(embedded, hidden)
        # u_pred is computed here instead of in the coverage attention, because the
        # weight U_pred is shared and the coverage attention does not use pred for
        # anything else. This avoids computing it twice.
        u_pred = torch.matmul(self.U_pred, pred.squeeze(1).t()).t()
        context_low = self.coverage_attn_low(low_res, u_pred)
        context_high = self.coverage_attn_high(high_res, u_pred)
        context = torch.cat((context_low, context_high), dim=1)
        new_hidden, _ = self.gru2(context.unsqueeze(1), pred.transpose(0, 1))
        w_s = torch.matmul(self.W_s, new_hidden.squeeze(1).t()).t()
        w_c = torch.matmul(self.W_c, context.t()).t()
        out = embedded.squeeze(1) + w_s + w_c
        out = self.maxout(out)
        out = torch.matmul(self.W_o, out.t()).t()
        return out, new_hidden.transpose(0, 1)
