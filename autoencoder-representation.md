## Autoencoders: Learning by Copying

### Introduction

Autoencoders (self-encoders) are machine learning models that attempt to replicate the input in their output. This endeavor may not seem to be very useful for tasks such as generating images or learning how inputs are related to one another, particularly when the models used are very large and fully capable of learning the identity function on the input.  Indeed, in the past it was assumed that models capable of learning the identity function (typically models with more neurons per layer than input elements) would do so without other forms of regularization.

Historically, therefore, autoencoders were studied in two contexts: as undercomplete models that compressed the input in some useful way, and as possibly overcomplete models that were subject to certain restrictions to as to prevent the learning of the identity function.  Here 'overcomplete' is used more informally than the strict linear algebra term, and signifies a model that has more than enough elements (neurons) per layer to exactly copy the input onto that layer.

A number of useful models have resulted from those efforts, including the contractive autoencoder which places a penalty on the norm of the Jacobian of the encoder with respect to the input, and the variational autoencoder which simultaneously seeks to minimize the difference between autoencoder output and input together with the difference between the hidden layer code $z$ given model inputs and a Gaussian distribution $N(z; I)$.

The primary challenge of these regularized autoencoders is that they tend to be incapable of modeling the distribution of inputs given to the model to a sufficient degree, which manifests experimentally as blurriness in samples drawn from these models. For variational autoencoders, there is a clear reason why samples would be blurry: transforming a typical image using a Gaussian distribution (via convolution or a similar method) results in a blurry copy of that image, so it is little surprise that enforcing the encoding of an input to approximate a Gaussian distribution would have the same effect.  As a consequence of this challenge, much research on generative models (particularly for images) has shifted away to [generative adversarial networks](https://blbadger.github.io/generative-adversarials.html) and more recently [diffusion models](https://blbadger.github.io/generative-adversarials.html).

Recently, however, two findings have brought the original assumption that overcomplete autoencoders would memorize their inputs into question. First, it has been observed that deep learning models for image classification tend to lose a substantial amount of input information by the time the input has been passed to deep layers, and that trained models learn to infer this lost information ([reference](https://arxiv.org/abs/2211.06496)).  Second it has been found that image classification models are more or less guaranteed to generalize to some extent when trained using gradient descent ([reference](https://arxiv.org/abs/2211.09639)).

In light of these findings, it is worth asking whether overcomplete deep learning autoencoders will necessarily memorize their inputs.  The answer to this question is straightforward, as we will see in the next section.

### Overcomplete autoencoders do not typically memorize when trained via gradient descent

The question of whether overcomplete autoencoders memorize their inputs can easily be tested by first training an autoencoder to replicate inputs, and then observing the tendancy of that autoencoder to copy its inputs after small changes are made.  Autoencoders may be trained in various ways, and we choose to employ the minimization of Mean-Squared Error between the input $a$ and reconstructed input $D(E(a))$ where $D$ signifies the decoder of hidden code $h$, $E$ the encoder which maps $a \to h$ and $O(a, \theta)$ the output of the encoder and decoder given model parameters $\theta$.

$$
L(O(a, \theta), a) = || D(E(a)) - a ||^2_2
$$

Mean-squared error loss in this case is equivalent to the cross entropy between the model's outputs and a Gaussian mixture model of those outputs.  More precisely assuming that the distribution of targets $y$ given inputs $a$ is Gaussian, the likelihood of $y$ given input $a$ is

$$
p(y | a) = \mathcal{N}(y; \mu (O(a, \theta)), \sigma)
$$

where $\mu (O(a, \theta))$ signifies the mean of the output and $y=a$ for an autoencoder.

As for the case observed with variational autoencoders, it has been assumed that training autoencoders via MSE loss results in blurry output images because it implies that the input can be modelled by a Gaussian mixture model.  And for a sufficiently small model this is true, being that only a small number of Gaussian distributions may compose the mixture model.  
But for a very large model it is not the case that MSE loss (or any maximum likelihood estimation approach) will necessarily yield blurry outputs because the Gaussian mixture model is a universal approximator of computable functions if given enough individual Gaussian distributions independent from each other given weights $w_1, w_2, ... w_n$

$$
g(a; w, \mu, \sigma) = \sum_i w_i \mathcal{N}(a; \mu_i, \sigma_i)
$$

Therefore it can be theoretically guaranteed that a sufficiently large autoencoder trained using MSE loss on the output alone will be capable of arbitrarily good approximations on a given input $a$ (and will not yield blurry samples).  

This being the case, we can now test whether an overcomplete autoencoder trained using gradient descent to minimize MSE loss will memorize their inputs.  Memorization in this case is equivalent to learning the identity function, so we can test memorization by simply changing an input some small amount before observing the output.  If the identity function has been learned then the model will simply yield the same input back again.

One such change to an input is the addition of noise.  In the following experiment we train a 5-hidden layer feedforward model, each layer having more elements that the input. As shown in the figure below, we find that indeed the overcomplete autoencoder does not memorize its inputs: upon the addition of Gaussian noise $\mathcal N(a \in [0, 1]; \mu=7/10, \sigma=2/10)$

![overcomplete autoencoder]({{https://blbadger.github.io}}/deep-learning/overcomplete_cifar10_autoencoder.png)

This experiment provides evidence for two ideas: first that overcomplete autoencoders do not tend to learn the identity function as evidenced by the difference in the autoencoder's output given a noise-corrupted input, $O(\widehat a, \theta)$ compared to the original input $a$, and second that the output actually de-noises the input such that 

$$
||O(\widehat a, \theta) - a|| < || \widehat a - a ||
$$

The latter observation is far from trivial: there is no guarantee that a function learned by an autoencoder, even if this function is not the identity, would be capable of de-noising an input. We next investigate why this would occur.

### Autoencoders are capable of denoising an input without being explicitly training to do so

To re-iterate, [elsewhere](https://blbadger.github.io/depth-generality.html) it was found that deep learning models lose a substantial amount of information about the input in deep layers and tend to learn to infer that lost information upon training.  The learning process results in arbitrary inputs being mapped to the learned manifold in the layers preceeding.

This process may be thought of as analagous to learning how to de-noise an input,

For example, take the last layer's representations of the input in trained versus untrained Unet model (wihtout residual layers) shown in the following figure.

![landscape representations]({{https://blbadger.github.io}}/deep-learning/unet_landscape_representations.png)

The trained model has clearly learned to reduce the noise in the last layer's representation relative to the untrained model. It may thus be wondered whether autoencoders are capable of learning to de-noise inputs.

![denoising autoencoder]({{https://blbadger.github.io}}/deep-learning/unet_512_denoising.png)

The same ability of autoencoders to de-noise is observed for Unet applied as an autoencoder for lower-resolution images, here 64x64 LSUN church images.

![overcomplete autoencoder]({{https://blbadger.github.io}}/deep-learning/unet_autoencoding_churches.png)

![overcomplete autoencoder]({{https://blbadger.github.io}}/deep-learning/unet_autoencoding_churches_2.png)

### Image Generation with an Autoencoder Manifold Walk

So far we have seen that autoencoders are capable of removing noise from an input even when they are not trianed to do so, and rather are tasked with copying the input to the output.  There is a clear theoretical basis for why this would occur: approximate and absolute non-invertibility between subsequent layers of (even very large and overcomplete) autoencoders necessarily introduces noise into the deep layer's representations of the input.

Noise is introduced between subsequenty non-invertible (or approximately non-invertible) layers for the simple reason that many possible inputs to that layer may yield one identical output.  These many possible inputs resemble Gaussian noise if if the layer in question contains a large number of independent elements, which is indeed the case for the distributed representations that compose deep learning models. 

To see why non-invertibility for a transformation consisting of a large number of independent elements results in the introduction of Gaussian noise in the output's representation of the input, first define the transformation of some layer of our model to be a function $f: \Bbb R^n \to \Bbb R^m$ where the input $a$ contains $n$ elements and the output $m$ elements. For a fully connected feedforward deep learning model, $f$ is a composition of $m$ functions $f_1, f_2, ... f_m$ each taking all $n$ elements as their inputs as follows:

$$
f = \{ f_1(a_1, a_2, ..., a_n), f_2(a_1, a_2, ... a_n), ... ,f_m(a_1, a_2, ..., a_n)\}
$$

where each $f_m$ is typically a degree one polynomial of the input,

$$
f_m(a_1, a_2, ..., a_n) = w_{1, m} a_1 + w_{2, m} a_2 + \cdots + w_{n, m} a_n
$$

Now consider the significance of the output of $f$ being non-unique with respect to the input: this means that many different vectors $a$ yields some identical $f(a)$.  Without prior knowledge besides the given values of $w$ which define the output $f(a)$, we can therefore form $m$ probability distributions describing the likelihood of all possible vectors of $a$ that satisfy $f_m^{-1}(f(a))$.  

Recalling that the output $f_m$ is a polynomial of all elements of $a_n$, we can express $f_m$ as follows

$$
f_m(a) = w_{1, m} * p(a_1 | f(a)) + w_{2, m} * p(a_2 | f(a)) + \cdots + w_{n, m} * p(a_n | f(a))
$$

where $p(a_n \vert f(a))$ signifies the probability of input element $a_n$ given the output $f(a)$. We can draw random variable samples from these probability distributions, $A_i \sim p(a_i \vert f(a))$ such that $f_m(a)$ can be expressed as

$$
f_m(a) = w_{1, m} * A_1 + w_{2, m} * A_2 + \cdots + w_{n, m} * A_n
$$

Ignoring for the present approximate non-invertibility, for absolute non-invertibility only each distribution $p(a_n)$ is uniform over all valid inputs that satisfy $\{ a_{n, i} : $f_m^{-1}(f(a_n)) = a_n \}$ so that the probability distributions are as follows:

$$
p(a_n | f(a)) = \mathcal{U} (a_{n, i})
$$

Considering approximate non-invertiblity once again, the distributions $p(a_n \vert f(a))$ are typically not uniform and are indeed  difficult to express exactly.  The exact nature of each distribution is irrelevant, however, as we can apply the central limit theorem to $f_m(a)$ because it is the sum of many independent variables.

This is significant because the central limit theorem states that addition of many independent distributions (of any identity) tends towards a Gaussian distribution.  Therefore the addition of any set of distributions of possible values of $p(a_1)$ tends towards the Gausian distribution as $n \to \infty$. 

One note on the preceeding argument: it may at first seem absurd to suppose that $A_1, A_2, A_3, ..., A_n$ are independent random variables because if one chooses a certain value for the first element of $a_1 = A_1$ then there is no reason to suppose that this does not limit the possibilities for choosing subsequent values $a_2, a_3, ..., a_n$.  But this does not mean that random variables $A_1, A_2, A_3, ..., A_n$ are dependent because the choosing of one random variable from $ A_n \sim p(a_n \vert f(a))$ does not affect which element is chosen from any other distribution, rather only that the joint distribution $p(A_1, A_2, ..., A_n)$ is extremely difficult to explicitly describe.   

{% include youtube.html id='SzzIJD05aVI' %}

### Representations in Unet generative models with attention

It may be wondered why representation clarity (or lack thereof) would be significant.  So far we have seen that the more accurate (ie Vit Large 16) vision transforms contain representations in their hidden layers that mirror those of convolutional models such as ResNet50, in which a near-perfect autoencoding of the input may be made without prior knowledge in early layers but a substantial amount of information is lost during the forward pass, leading to poor input representation accuracy in later layers.

For tasks of classification, the importance of accurate input representation is unclear.  One might assume that a poor representation accuracy in deep layers prevents overfitting, but as observed [here](https://arxiv.org/pdf/2211.09639.pdf) the gradient descent process itself serves to prevent overfitting regardless of representational accuracy for classifiers.  For generative models, however, there is a theoretical basis as to why imperfect deep representation is favorable.

To see why perfect input representation in the output may be detrimental to generative models, take a simple autoencoder designed to copy the input to the output.  For a useful autoencoder we need the model $\theta$ to learn parameters that capture some important invariants in the data distribution $p(x)$ of interest.  But with perfect input representation capability in the output the model may achieve arbitrarily low cost function value by learning to simply copy the input to output.

Next consider the case of a denoising autoencoder.  This model is tasked with recoving the input $x$ drawn from the distribution $p(x)$ followed by a corruption process $C(x) = \tilde{x}$, minimizing the likelihood that the model output $ O(\tilde{x}, \theta) $ was not drawn from $p(x)$.  Under some assumptions this is equivalent to minimizing the mean squared error 

$$
L = || O(\tilde{x}, \theta) - x ||_2^2
$$

Now consider the effect of a perfect representation of the input $\tilde{x}$ in the output with respect to minimizing $L$.  Then we have

$$
L = || \tilde{x} - x ||_2^2
$$

which is the MSE distance taken by the corruption process. Therefore the model is capable of inferring the corruption process without training.


### Autoencoder latent space
