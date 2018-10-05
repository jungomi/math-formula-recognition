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


class BottleneckBlock(nn.Module):
    def __init__(self, input_size, growth_rate):
        super(BottleneckBlock, self).__init__()
        inter_size = num_bn * growth_rate
        self.norm1 = nn.BatchNorm2d(input_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_size, inter_size,
                               kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(inter_size)
        self.conv2 = nn.Conv2d(inter_size, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu(self.norm1(x)))
        out = self.conv2(self.relu(self.norm2(out)))
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(TransitionBlock, self).__init__()
        self.norm = nn.BatchNorm2d(input_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(input_size, output_size,
                              kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(self.relu(self.norm(x)))
        return self.pool(out)


class DenseBlock(nn.Module):
    def __init__(self, input_size, growth_rate, depth):
        super(DenseBlock, self).__init__()
        layers = [BottleneckBlock(input_size + i * growth_rate, growth_rate)
                  for i in range(depth)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, num_in_features=48):
        super(Encoder, self).__init__()
        self.conv0 = nn.Conv2d(
            3, num_in_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm0 = nn.BatchNorm2d(num_in_features)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        num_features = num_in_features
        self.block1 = DenseBlock(
            num_features, growth_rate=growth_rate, depth=depth)
        num_features = num_features + depth * growth_rate
        self.trans1 = TransitionBlock(num_features, num_features // 2)
        num_features = num_features // 2
        self.block2 = DenseBlock(
            num_features, growth_rate=growth_rate, depth=depth)

        num_features = num_features + depth * growth_rate
        self.trans2_norm = nn.BatchNorm2d(num_features)
        self.trans2_relu = nn.ReLU(inplace=True)
        self.trans2_conv = nn.Conv2d(num_features, num_features // 2,
                                     kernel_size=1, stride=1, bias=False)
        self.trans2_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.multi_block = DenseBlock(
            num_features, growth_rate=growth_rate, depth=multi_block_depth)
        num_features = num_features // 2
        self.block3 = DenseBlock(
            num_features, growth_rate=growth_rate, depth=depth)
        num_features = num_features + depth * growth_rate // 2

    def forward(self, x):
        out = self.relu(self.norm0(self.conv0(x)))
        out = self.max_pool(out)
        out = self.block1(out)
        out = self.trans1(out)
        out = self.block2(out)
        out_before_trans2 = self.trans2_relu(self.trans2_norm(out))
        out_A = self.trans2_conv(out_before_trans2)
        out_A = self.trans2_pool(out_A)
        out_A = self.block3(out_A)
        out_B = self.multi_block(out_before_trans2)

        return out_A, out_B


class CoverageAttention(nn.Module):
    # input_size = C
    # output_size = q
    # attn_size = L = H * W
    def __init__(self, input_size, output_size, attn_size, kernel_size,
                 device=device):
        super(CoverageAttention, self).__init__()
        self.alpha = torch.zeros((1, attn_size), device=device)
        self.conv = nn.Conv2d(input_size, output_size, kernel_size=kernel_size)
        self.fc = nn.Linear(attn_size, attn_size * output_size)
        self.U_pred = nn.Parameter(torch.randn((n_prime, n)))
        self.U_a = nn.Parameter(torch.randn((n_prime, input_size)))
        self.U_f = nn.Parameter(torch.randn((n_prime, output_size)))
        self.nu_attn = nn.Parameter(torch.randn(n_prime))
        self.input_size = input_size
        self.output_size = output_size
        self.attn_size = attn_size
        self.device = device

    def reset_alpha(self):
        self.alpha = torch.zeros((1, self.attn_size), device=self.device)

    def forward(self, x, pred):
        batch_size = x.size(0)
        # TODO: Use the convolutional layer.
        # The linear layer is just to make it work until I figure out the Conv.
        out_f = self.fc(self.alpha.sum(0)).view(
            1, -1, self.output_size).expand(batch_size, -1, -1)
        # Get rid of seq_len (dim 1, which is always 1)
        # Transpose to get input_size x batch_size to multiply
        pred_view = pred.squeeze(1).t()
        u_pred = torch.matmul(self.U_pred, pred_view)
        # Transpose back to get batch_size x n_prime
        u_pred = u_pred.t()
        # Change the dimensions
        # From: (batch_size x C x H x W)
        # To: (batch_size x C x L)
        a = x.view(batch_size, x.size(1), -1)
        u_a = torch.matmul(self.U_a, a)
        u_f = torch.matmul(self.U_f, out_f.transpose(1, 2))
        # u_pred is expanded from (batch_size x n_prime)
        # to (batch_size x n_prime x L) because there are L components to which
        # the same u_pred is added.
        u_pred_expanded = u_pred.unsqueeze(2).expand_as(u_a)
        tan_res = torch.tanh(u_pred_expanded + u_a + u_f)
        e_t = torch.matmul(self.nu_attn, tan_res)
        alpha_t = torch.softmax(e_t, dim=1)
        self.alpha = torch.cat((self.alpha, alpha_t), dim=0)
        # alpha_t: (batch_size x L)
        # a: (batch_size x C x L) but need (C x batch_size x L) for
        # element-wise multiplication. So transpose them.
        cA_t_L = alpha_t * a.transpose(0, 1)
        # Transpose back
        return cA_t_L.transpose(0, 1).sum(2)


class Decoder(nn.Module):
    def __init__(self, num_classes, low_res_shape, high_res_shape,
                 hidden_size=gru_hidden_size, embedding_dim=embedding_dim,
                 device=device):
        super(Decoder, self).__init__()
        C = low_res_shape[0]
        C_prime = high_res_shape[0]
        context_size = C + C_prime
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.gru1 = nn.GRU(input_size=embedding_dim,
                           hidden_size=hidden_size, batch_first=True)
        self.gru2 = nn.GRU(input_size=context_size,
                           hidden_size=hidden_size, batch_first=True)
        # L = H * W
        low_res_attn_size = low_res_shape[1] * low_res_shape[2]
        high_res_attn_size = high_res_shape[1] * high_res_shape[2]
        self.coverage_attn_low = CoverageAttention(
            C, decoder_conv_filters,
            attn_size=low_res_attn_size, kernel_size=11, device=device)
        self.coverage_attn_high = CoverageAttention(
            C_prime, decoder_conv_filters,
            attn_size=high_res_attn_size, kernel_size=7, device=device)
        self.W_o = nn.Parameter(torch.randn((num_classes, embedding_dim)))
        self.W_s = nn.Parameter(torch.randn((embedding_dim, hidden_size)))
        self.W_c = nn.Parameter(torch.randn((embedding_dim, context_size)))
        self.hidden_size = hidden_size

    def init_hidden(self, batch_size):
        return torch.zeros((1, batch_size, self.hidden_size))

    def reset(self):
        self.coverage_attn_low.reset_alpha()
        self.coverage_attn_high.reset_alpha()

    # TODO: Figure out what to do with the new hidden state returned from the
    # GRUs. Apparently they aren't kept, since the new hidden state of the
    # decoder is the output of the second GRU.
    #
    # Unsqueeze and squeeze are used to add and remove the seq_len dimension,
    # which is always 1 since there only the previous symbol is provided, not
    # a sequence.
    # The inputs that are multiplied by the weights are transposed to get
    # (m x batch_size) instead of (batch_size x m). The result of the
    # multiplication is tranposed back.
    def forward(self, x, hidden, low_res, high_res):
        embedded = self.embedding(x)
        pred, _ = self.gru1(embedded, hidden)
        context_low = self.coverage_attn_low(low_res, pred)
        context_high = self.coverage_attn_high(high_res, pred)
        context = torch.cat((context_low, context_high), dim=1)
        new_hidden, _ = self.gru2(context.unsqueeze(1), pred.transpose(0, 1))
        w_s = torch.matmul(self.W_s, new_hidden.squeeze(1).t()).t()
        w_c = torch.matmul(self.W_c, context.t()).t()
        # TODO: Maxout this
        out = embedded.squeeze(1) + w_s + w_c
        out = torch.matmul(self.W_o, out.t()).t()
        return out, new_hidden.transpose(0, 1)
