# First implementation

I implemented the paper [Multi-Scale Attention with Dense Encoder for Handwritten Mathematical Expression Recognition][arxiv-zhang18].
There are certain parts that aren't completely clear, mostly because of lack of
information, but at least it works out and seems fairly logical.

## Table of contents

<!-- vim-markdown-toc GFM -->

* [Dataset](#dataset)
* [Details of the model's architecture](#details-of-the-models-architecture)
  * [Encoder](#encoder)
  * [Decoder](#decoder)
    * [Coverage attention](#coverage-attention)
    * [Dimensions](#dimensions)
    * [Convolution Q](#convolution-q)
    * [Maxout activation function *h*](#maxout-activation-function-h)
* [Needs discussion / clarification](#needs-discussion--clarification)

<!-- vim-markdown-toc -->

## Dataset

[CROHME: Competition on Recognition of Online Handwritten Mathematical Expressions][crohme] has been used.

The paper focuses on off-line recognition. Hence, I've converted the dataset to
images of size `256x256` and extracted the ground truth for off-line recognition.
The converted dataset can be found at [Floydhub - crohme-png][crohme-png].

## Details of the model's architecture

### Encoder

It's based on [DenseNet][arxiv-densenet], but adds a second branch before the
last pooling layer of the regular branch.

![Dense Encoder architecture][encoder-architecture]

Branch A produces the low resolution annotations, which is the output
of a regular DenseNet, and branch B produces the high resolution annotations.

```python
low_res = (batch_size, C, H, W)
high_res = (batch_size, C_prime, 2H, 2W)
```

Example:

```python
# Features created by the first convolutional filter before any dense block
num_in_features = 48
image_size = (256, 256)

low_res = (batch_size, 684, 16, 16)
high_res = (batch_size, 792, 32, 32)
```

These features can be seen as annotations of length `L = H x W` and
`4L = 2H x 2W` respectively.

A = { a<sub>1</sub>, ..., a<sub>L</sub> } and
B = { b<sub>1</sub>, ..., b<sub>4L</sub> }

Therefore the dimensions are `A = (L x C)` and `B = (L x C_prime)`.

In Pytorch the channels are usually first, which is beneficial because it would
later need to be transposed in order to multiply them by the weight in the
decoder.

```python
A = (batch_size, C, L)
B = (batch_size, C_prime, 4L)
```

### Decoder

The decoder uses two Gated Recurrent Units (GRU) with a coverage attention. The
first GRU receives the previous symbol and the previous hidden state of the
decoder and produces a prediction of the next symbol.

*pred<sub>t</sub>* = GRU(y<sub>t-1</sub>, *hidden*<sub>t-1</sub>)

The second GRU creates the next hidden state of the decoder from the context
vector c<sub>t</sub> and the prediction (output of the first GRU).

*hidden<sub>t</sub>* = GRU(c<sub>t</sub>, *pred*<sub>t</sub>)

*y<sub>t</sub>* = W<sub>o</sub>*h*(E y<sub>t-1</sub>
    + W<sub>pred</sub> *pred*<sub>t</sub> + W<sub>c</sub>C<sub>t</sub>)

E y<sub>t-1</sub> is the previous symbol's embedding.

`h` is supposed to be a maxout activation function.

#### Coverage attention

There are two separate coverage attention models, which compute the
low resolution and high resolution context vectors. The final context vector is
the concatenation of both.

*e<sub>ti</sub>* = 𝛎<sub>att</sub><sup>T</sup> tanh(U<sub>pred</sub> *pred*<sub>t</sub>
    + U<sub>a</sub>a<sub>i</sub> + U<sub>f</sub>f<sub>i</sub>)

*α<sub>t</sub>* = Softmax(e<sub>t</sub>)

*F* = Q * (Σ α<sub>t</sub>)

*cA<sub>t</sub>* = Σ (α<sub>t</sub> * a)

Where `Q` is a convolutional layer with `q` output channels.

*result*<sub>u_pred</sub> = U<sub>pred</sub> *pred*<sub>t</sub> - can be
computed outside the coverage attention because it remains the same for both
models.

*e<sub>t</sub>* = 𝛎<sub>att</sub><sup>T</sup> tanh(*expanded*<sub>u_pred</sub>
    + U<sub>a</sub>a + U<sub>f</sub>F)

with *expanded*<sub>u_pred</sub> = [r<sub>1</sub>, ..., r<sub>L</sub>],
r<sub>i</sub> = *result*<sub>u_pred</sub> ∀i ∈ [1, L]

It is expanded to vectorise the addition.

#### Dimensions

Mostly writing this down here, because I've used it to get the implementation
details right and figure out unknown parts (e.g. `Q`). As long as the dimensions
fit, it can't be that wrong.

```python
# Embedding dimension
m = 256
# Decoder hidden dimension
n = 256
# Attention dimension
n_prime = 512
# Output channels of convolution Q
q = 256
# Low resolution features
C = 684
# High resolution features
C_prime = 792

U_pred = (n_prime, n)
U_a = (n_prime, C)
U_b = (n_prime, C_prime)
U_f = (n_prime, q)

pred = (batch_size, n)
a = (batch_size, C, L)
b = (batch_size, C_prime, 4L)

# The dimensions below are for the case of a, but equally apply to b
# by replacing C with C_prime and L with 4L
F = (batch_size, q, L)

result_u_pred = (batch_size, n_prime)
expanded_u_pred = (batch_size, n_prime, L)
result_u_a = (batch_size, n_prime, L)
result_u_f = (batch_size, n_prime, L)

𝛎_att = (n_prime)
e_t = (batch_size, L)
α_t = (batch_size, L)

cA_t = (batch_size, C)
cB_t = (batch_size, C_prime)
c_t = (batch_size, C + C_prime)


W_o = (vocab_size, m)
W_pred = (m, n)
W_c = (m, C + C_prime)

embedded_prev = (batch_size, m)
result_w_pred = (batch_size, m)
result_w_c = (batch_size, m)

result_w_o = (batch_size, vocab_size)
```

#### Convolution Q

Since α<sub>t</sub> has dimension `(batch_size, L)` the input of the convolution
`Q`. In the paper [Track, Attend and Parse (TAP): An End-to-end Framework for Online Handwritten Mathematical Expression Recognition][tap],
it was mentioned that `Q` is a 1D convolution. To make it work a sequence length
of 1 is introduced, but because the output must be of dimension
`(batch_size, q, L)` it requires a padding to not lose any width. (e.g. for
a kernel size of 11 a padding of 5 is needed)

In this paper however, it is said to be an `11x11` and `7x7` kernel size for the
low resolution and high resolution convolution respectively, which means it must
be a 2D convolution. Since `L = H x W`, it is possible to reshape α<sub>t</sub>
to `(batch_size, H, W)`. The padding is still needed for the same reason, i.e.
the kernel size `11x11` requires a padding of 5 and the kernel size `7x7`
requires a padding of 3.

#### Maxout activation function *h*

Maxout for me is the maxout layer/neuron *max*(w<sub>1</sub><sup>T</sup>x + b<sub>1</sub>, w<sub>2</sub><sup>T</sup>x + b<sub>2</sub>).
Looking for maxout in Pytorch I found: [Pytorch - Maxout layer][maxout-issue],
which splits up the last dimension into pools of elements and then only keeps
the maximum from each pool. With a pool size of 2, the total size would be
halved. That is definitely better than introducing even more parameters with the
above layer.

As W<sub>o</sub> is supposedly of dimension `(vocab_size, m/2)` it would make
sense to have a maxout of 2.

## Needs discussion / clarification

- **Convolution `Q`**: Not entirely sure about it, but so far the implementation
  seems to fit in nicely.
- **Maxout `h`**: Looks like some regularisation at the end, but reducing the output
  dimension by half seems arbitrary, maybe there is a reason for it or it could
  be improved. It also wasn't present in their previous papers with similar
  approaches (or even identical for that part). I would at least have expected
  mentioning a reason for it.
- **Evaluation**: They mentioned having trained 5 models with different initialised
  parameters and used the average of their prediction probabilities during the
  evaluation. They compared models with different depth for the dense block of
  the multi-scale branch, they may have used these, otherwise there are
  plenty of parameters that could be changed.
- **Dataset**: The ground truths don't seem consistent within the training set,
  mainly missing curly brackets for example with `\frac` it's sometimes
  `\frac 2 3` instead of `\frac{2}{3}`. It looks like it's an oddity of the
  `KME*.png` samples from the `KAIST` set. Should I change these? Well, sadly
  it is valid LaTeX... (I guess the answer is no)
- **Word Error Rate (WER)**: They also used the WER metric besides the official
  CROHME evaluating metric (expression recognition rates). I'm struggling to
  clearly define a *word* in a LaTeX formula, e.g. is `F_{x}` one word? two? or
  more? Apparently they used it at a symbol level, which makes it more
  reasonable, but you can hardly call that *Word* Error Rate, because symbols
  are pretty much equivalent to characters in natural language. In `F_{x}` there
  are already 5 symbols.

[arxiv-zhang18]: https://arxiv.org/pdf/1801.03530.pdf
[arxiv-densenet]: https://arxiv.org/pdf/1608.06993.pdf
[code-repo]: https://github.com/jungomi/math-formula-recognition
[crohme]: https://www.isical.ac.in/~crohme/
[crohme-png]: https://www.floydhub.com/jungomi/datasets/crohme-png
[encoder-architecture]: ./figures/dense-encoder.png
[maxout-issue]: https://github.com/pytorch/pytorch/issues/805#issuecomment-389447728
[tap]: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8373726
