## Masked Recurrent Mixers

The most efficient non-trivial language model for autoregressive inference are models that read the current token, load a fixed memory representing previous tokens, and predict a next token. This is essentially how recurrent neural networks perform inference, such that any model that may be inferenced using this property may be thought of as a kind of RNN. In the past, it was observed that training RNNs requires a reformulation of the inference process to make use of parallelization (or each forward pass may only train using the output of a single next token, rather than all next tokens in a sequence), and that this training is beset by a number of problems once this is done (numerical stability from back-propegation through time, vanishing gradients, slow convergence etc.) such that current large language model architectures that load a variable amount of memory (for example transformers) have provem much more powerful for most tasks, despite their lower inference efficiencies. If we think about this in terms of inference computational efficiency versus training efficiency and task accuracy, it seems that defining a model first by its inference characteristics and then attempting to increase its training and task performance has proven difficult.

On this page we explore an inversion of this approach: we instead take architectures that we know are relatively efficient to train and capable of many useful language tasks and attempt to constrain them to be efficient to inference, while reducing their training and task performance as little as possible. This approach has been followed with varying degrees of success (in particular see Linear Transformers, linear state space models, and Mamba), but the hypothesis is that the [masked mixer](https://arxiv.org/abs/2409.01482) provides a better starting architecture with which to enforce constraints than the existing alternatives. It is helpful to first consider why this would be, and we do so in the next section, before proceeding to how this may be done and observing the results of doing so.

### Background

Our first hypothesis is as follows: adapting mixer layers for linear time and constant space complexity sequence modeling should be much easier than doing so for self-attention or state spaces. There are a few reasons why this is expected to be the case: firstly because masked mixers have been found to be efficiently trainable with only a single linear transformation (ie matrix multiplication) between tokens and constraining this transformation to use only $n$ rather than $n^2$ unique values as a Toeplitz matrix does not substantially affect training efficiency, secondly because these inter-token transformations are explicitly rather than implicitly defined on the input dataset, and finally because mixers generally exhibit superior information retention to other model types.

### Linear time and Constant Space Complexity Mixer Constraints



### What Token Mixing Weights do Masked Mixers Learn?

As previously mentioned, one substantial benefit of using masked mixers compared to transformers as a starting architecture for linear-complexity modeling is that the former use explicit parameterizations of inter-token transformations, whereas the latter use implicit parameterizations. What this means is that the inter-token transformations in masked mixers are, once trained, fixed and constant for all possible inputs, whereas these transformations are in effect defined by the data itself for transformers.

![masked mixer weight overview](/figures/masked_mixer_weight_fig.png)


![masked mixer weight overview](/figures/masked_mixer_fig2.png)










