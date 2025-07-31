### Linear Language Models

### Background

In the field of numerical analysis one can generally say that there are a number of differences between linear and nonlinear processes, at least at a very high level. Perhaps most notably, linear transformations may be completed in one effective 'step' whereas nonlinear transformations require many 'steps'. Whether this is accurate or not in practice is somewhat dubious, but for our particular application it will indeed be.

This is relevant because we can get an idea of how to make an extremely fast (to run, that is) language model by considering what exactly happens during autoregressive inference. When one considers autoregressive inference, it is generally noted that models like Transformers that compare all tokens to all other tokens scale with $n^3d$ time complexity without caching, and $n^2d$ with caching for $n$ tokens and hidden dimension $d$. It is less appreciated that inference time also depends on the number of layers of linear transformations in a model $l$, as because typically each layer is separated from each next layer by one or more nonlinear transformation (layer normalization, ReLU, GeLU etc.) such that the actual time complexity becomes $n^2dl^2$ as each of the $n^2$ token comparisons require $l$ steps. 

Clearly it would be advantageous to reduce the number of layers in a model, but how can this be done while maintaining an efficiently trainable architecture? To start to answer this question, one can instead ask the following: ignoring trainability, what is the minimum number of layers in a causal language model? The answer is straightforward: a one-layer model contains the fewest layers, and a one-layer model in this case is equivalently a fully linear model. The question then becomes whether or not a linear or nearly-linear model is capable of being trained effectively, either on a small dataset like TinyStories or a larger dataset like the Fineweb.

For small datasets like TinyStories, the answer is somewhat surprisingly yes: fully linear language models are capable of generating gramatically correct text, although is a good deal less efficient to train than a nonlinear model.

### Linear Mixer Architecture

Before proceeding further, however, it is best to understand a few theoretical arguments for and against the use of linear models. The arguments for mostly revolve around their utility: they are fast (because they can be mostly or wholly parallelized on hardware), easy to optimize, and somewhat more interpretable than nonlinear ones. The downsides revolve around their lack of representational power.

How might one go about converting a masked mixer into a linear model? We will take the approach to remove nonlinear transformations and optimize the resulting linear model to train most efficiently using adaptive versions of stochastic gradient descent. We start with a [masked mixer](https://blbadger.github.io/smaller-lms.html) as this model may be thought of as a transformer with self-attention replaced by a single linear transformation (a masked convolution and proceed to linearize the rest of the model.

Besides removing nonlinear activations, we must also deal with layer normalizations, which are defined as follows: for a specific layer input $x$, the output $y$ is

$$
y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} * \gamma + \beta
$$

where $\gamma$ and $\beta$ are trainable parameters (which are usually held in very high precision) and $\mu, \sigma^2$ denote the mean (expectation) and variance of the random variable $x$. This is clearly a nonlinear operation (although it could be transformed into a linear one) and so we will simply remove this transformation for now.

We also remove superfluous linear transformations: in particular the feedforward module of a transformer usually contains a up projection $U$, nonlinear activation, and down projection $D$. Without the nonlinearity, we are left with a sequence of transformations applied to input $x$ expressed as $DUx$, but we can compose these two linear transformations into one, ie $DUx=Wx$, for efficiency. Furthermore, we can also compose this feedforward weight matrix with the language modeling head layer $H_{lm head}$, as $HWx = Qx$ for all matrices $H, W$ and some $Q$. We could go further and continue to compose layers together, but as we will see in the next section this is surprisingly not always benefical for efficient training.

The linear masked mixer architecture we use may be compared with the transformer and masked mixer diagramatically. In the following depiction, all nonlinear transformations are colored red and all linear transformations are blue or purple.

![linear mixer arch](/deep-learning/linear_mixer_architecture.png)

### Larger linear model layer widths lead to more efficient training

We can be confident that a linear model will not have the representational power of a nonlinear one from an abundance of arguments for this idea. But can a linear model be trained to effectively model language? At least for simple langauge datasets such as TinyStories (synthetic paragraph-length stories written as if from a five-year-old's perspective, albeit much less creatively than what you would get from a real five-year-old), the answer is suprisingly yes as we will see below. This dataset is certainly much less challenging to model compared to broad text corpora with code, mathematics, web pages, and books that are commonly fed to frontier models today, but it is important to note that much is included in modeling even TinyStories; factual recall (names and places and simple events), grammar, story coherence and context for a few.

What is necessary for efficient TinyStories modeling? Some experimentation can convince use that simply stacking linear layers does not increase training efficiency, and indeed decreases it slightly. This is true both when the entire linear mixer architecture (above) contains more than one module, or else when we use two or three or more linear transformations after the convolutional layer. 

The above results are to be expected from the fundamentals of linear algebra where matrix multiplication is by definition linear such that multiplication by multiple matrices in succession may always be reduced to multiplication by a single matrix (assuming no nonlinearities are added between layers). This is true regardless of whether or not the weight matrices expand the vector's width at intermediate stages or not: for example if $W$ is a 8x2 (m by n, rows by columns) matrix and $H$ an 2x8 matrix such that $Wx$ expands $x$ by a factor of four and $H(Wx)$ reduces $Wx$ by a factor of four again then one can always make an equivalent matrix $Q$ that is 2x2. 

From the above argument, one would not expect for matrices with 'expanded' hidden layers in a linear model to be beneficial, and this is true for TinyStories modeling when applying feedforward layers. It is suprisingly not true for the model as a whole: given a tokenizer of size $n_t=4096$, we observe the following cross-entropy losses for linear mixers of various $d_m$ when trained on TinyStories:

| dm  | 4096  | 8192  | 16384  | 32768  |  65536 |
|---|---|---|---|---|---|
| 1e | 3.08  | 2.95  | 2.84  | 2.74  | 2.65 |
| 2e |  2.94 | 2.81  | 2.69  | 2.59  | 2.50 |
| 3e | 2.87  | 2.75  | 2.62  | 2.50  | 2.41 |

To show why this is suprising, it is helpful to map out the computational graph of a small $n_{ctx}=3$ linear mixer. In the following figure, we assume that the tokens have been pre-converted to one-hot vectors and denote vectors in lower-case italic letters, constants in lower-case letters, and matrices in upper-case letters.

![linear mixer computation](/deep-learning/linear_mixer_computation.png)

The equivalence in the top row to the second row may be seen as all convolutional weight elements are constants, and scaling a given matrices' row before multiplying to another matrices' column (ie forming a dot product) is equivalent to multiplying first and scaling after, and follows from the linearity of matrix multiplication. Now note that $HWx$ fulfills the same matrix multiplication we saw before: one should not expect to have any performance increase if $W$ expands the dimensionality of $x$ and $H$ decreases it, as one can always find a $Q$ that is equivalent that does not expand $x$. But we see above that increasing the hidden dimension (which is the expanded size of $x$ by $W$) actually vastly increases training efficiency.

For a 'nearly' linear mixer with inter-token transformations conv1D -> gelu -> conv1d (such that we only have $n_{ctx}$ nonlinear transformations in the model), we have the following:

| dm  | 4096  | 8192  | 16384  |
|---|---|---|---|
| 1e  | 3.25  | 3.27  | 3.24  |
|  2e |  3.09 | 3.04  | 3.00  |
|  3e | 3.02  | 2.94  | 2.87  |

Whereas for a one-layer mixer (with nonlinearities in each intra-token transformation) or transformer, we have

| dm  | Mixer 4096  | Mixer 8192  | Llama 4096 |
|---|---|---|---|
| 1e  | 2.51  | 2.38 |  2.30 |
|  2e |  2.36 | 2.25  |  2.18 |
|  3e | 2.28  | 2.16  |  2.12 |

This is somewhat promising: 

### 




