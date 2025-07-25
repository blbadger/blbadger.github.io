### Linear Language Models

### Background

In the field of numerical analysis one can generally say that there are a number of differences between linear and nonlinear processes, at least at a very high level. Perhaps most notably, linear transformations may be completed in one effective 'step' whereas nonlinear transformations require many 'steps'. Whether this is accurate or not in practice is somewhat dubious, but for our particular application it will indeed be.

This is relevant because we can get an idea of how to make an extremely fast (to run, that is) language model by considering what exactly happens during autoregressive inference. When one considers autoregressive inference, it is generally noted that models like Transformers that compare all tokens to all other tokens scale with $n^3d$ time complexity without caching, and $n^2d$ with caching for $n$ tokens and hidden dimension $d$. It is less appreciated that inference time also depends on the number of layers of linear transformations in a model $l$, as because typically each layer is separated from each next layer by one or more nonlinear transformation (layer normalization, ReLU, GeLU etc.) such that the actual time complexity becomes $n^2dl^2$ as each of the $n^2$ token comparisons require $l$ steps. 

Clearly it would be advantageous to reduce the number of layers in a model, but how can this be done while maintaining an efficiently trainable architecture? To start to answer this question, one can instead ask the following: ignoring trainability, what is the minimum number of layers in a causal language model? The answer is straightforward: a one-layer model contains the fewest layers, and a one-layer model in this case is equivalently a fully linear model. The question then becomes whether or not a linear or nearly-linear model is capable of being trained effectively, either on a small dataset like TinyStories or a larger dataset like the Fineweb.

For small datasets like TinyStories, the answer is somewhat surprisingly yes: fully linear language models are capable of generating gramatically correct text, although is a good deal less efficient to train than a nonlinear model.

### Linear Mixer Architecture

Before proceeding further, however, it is best to understand a few theoretical arguments for and against the use of linear models. The arguments for mostly revolve around their utility: they are fast (because they can be mostly or wholly parallelized on hardware), easy to optimize, and somewhat more interpretable than nonlinear ones. The downsides revolve around their lack of representational power: 

How might one go about converting a masked mixer into a linear model? We will take the approach to remove nonlinear transformations and optimize the resulting linear model to train most efficiently using adaptive versions of stochastic gradient descent.

The equation for layer normalization is as follows: for a specific layer input $x$, the output $y$ is

$$
y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} * \gamma + \beta
$$

where $\gamma$ and $\beta$ are trainable parameters (which are usually held in very high precision) and $\mu, \sigma^2$ denote the mean (expectation) and variance of the random variable $x$. This is clearly a nonlinear operation (although it could be transformed into a linear one) and so we will simply remove this transformation for now.

The architecture we use may be compared with the transformer and masked mixer diagramatically. In the following depiction, all nonlinear transformations are colored red and all linear transformations are blue or purple.

![linear mixer arch](/deep-learning/linear_mixer_architecture.png)


### TinyStories results

The results for a linear mixer are as follows:

| dm  | 4096  | 8192  | 16384  | 32768  |  65536 |
|---|---|---|---|---|---|
| 1e  | 3.08  | 2.95  | 2.84  | 2.74  | 2.65  |
|  2e |  2.94 | 2.81  | 2.69  | 2.59  | 2.50  |
|  3e | 2.87  | 2.75  | 2.62  | 2.50  | 2.41 |


For a 'nearly' linear mixer with inter-token transformations conv1D -> gelu -> conv1d, we have the following:

| dm  | 4096  | 8192  | 16384  |
|---|---|---|---|
| 1e  | 3.25  | 3.27  | 3.24  |
|  2e |  2.91 | 3.04  | 3.00  |
|  3e | 2.85  | 2.94  | 2.87  |

Whereas for a one-layer mixer (with nonlinearities in each intra-token transformation) or transformer, we have

| dm  | Mixer 4096  | Mixer 8192  | Llama 4096 |
|---|---|---|---|
| 1e  | 2.51  | 2.38 |  2.30 |
|  2e |  2.36 | 2.25  |  2.18 |
|  3e | 2.28  | 2.16  |  2.12 |

This is somewhat promising: 

### 




