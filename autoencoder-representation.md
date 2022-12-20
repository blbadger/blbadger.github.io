## Autoencoders: Learning by Copying

### Introduction

Autoencoders (self-encoders) are machine learning models that attempt to replicate the input in their output. This endeavor may not seem to be very useful for tasks such as generating images or learning how inputs are related to one another, particularly when the models used are very large and fully capable of learning the identity function on the input.




### Overcomplete autoencoders do not typically memorize when trained via gradient descent

In light of this finding, it may be wondered whether overcomplete autoencoders would actually learn an identity function as it has been assumed they would.  This can easily be tested by first training an overcomplete autoencoder to replicate inputs, and then observing the tendancy of that autoencoder to copy its inputs after small changes are made. One such change could be the addition of noise.  As shown in the figure below, we find that indeed an overcomplete autoencoder does not memorize its inputs.

![overcomplete autoencoder]({{https://blbadger.github.io}}/deep-learning/overcomplete-cifar10-autoencoder.png)


### Autoencoders are capable of denoising an input without being training to do so

To re-iterate, [elsewhere](https://blbadger.github.io/depth-generality.html) it has been found that deep learning models lose information about the input in deep layers and tend to learn to infer that lost information upon training.  This process may be thought of as analagous to learning how to de-noise an input: for example, take the representations learned by a Unet model (wihtout residual layers).

![landscape representations]({{https://blbadger.github.io}}/deep-learning/unet_landscape_representations.png)

The trained Unet has clearly learned to reduce the noise in the last layer's representation relative to the untrained model. It may thus be wondered whether autoencoders are capable of learning to de-noise inputs.

![denoising autoencoder]({{https://blbadger.github.io}}/deep-learning/unet_512_denoising.png)

The same is observed for Unet applied as an autoencoder for lower-resolution images, here 64x64 LSUN church images.

![overcomplete autoencoder]({{https://blbadger.github.io}}/deep-learning/unet_autoencoding_churches.png)

![overcomplete autoencoder]({{https://blbadger.github.io}}/deep-learning/unet_autoencoding_churches_2.png)

### Representations in Unet generative models with attention

It may be wondered why representation clarity (or lack thereof) would be significant.  So far we have seen that the more accurate (ie Vit Large 16) vision transforms contain representations in their hidden layers that mirror those of convolutional models such as ResNet50, in which a near-perfect autoencoding of the input may be made without prior knowledge in early layers but a substantial amount of information is lost during the forward pass, leading to poor input representation accuracy in later layers.

For tasks of classification, the importance of accurate input representation is unclear.  One might assume that a poor representation accuracy in deep layers prevents overfitting, but as observed [here](https://arxiv.org/pdf/2211.09639.pdf) the gradient descent process itself serves to prevent overfitting regardless of representational accuracy for classifiers.  For generative models, however, there is a theoretical basis as to why imperfect deep representation is favorable.

To see why perfect input representation in the output may be detrimental to generative models, take a simple autoencoder designed to copy the input to the output.  For a useful autoencoder we need the model $\theta$ to learn parameters that capture some important invariants in the data distribution $p(x)$ of interest.  But with perfect input representation capability in the output the model may achieve arbitrarily low cost function value by learning to simply copy the input to output.

Next consider the case of a denoising autoencoder.  This model is tasked with recoving the input $x$ drawn from the distribution $p(x)$ followed by a corruption process $C(x) = \tilde{x}$, minimizing the likelihood that the model output $ O(\tilde{x}, \theta) $ was not drawn from $p(x)$.  Under some assumptions this is equivalent to minimizing the mean squared error 

$$
L = || O(\tilde{x}, \theta) - x ||^2
$$

Now consider the effect of a perfect representation of the input $\tilde{x}$ in the output with respect to minimizing $L$.  Then we have

$$
L = || \tilde{x} - x ||^2
$$

which is the MSE distance taken by the corruption process. Therefore the model is capable of inferring the corruption process without training.


### Autoencoder latent space
