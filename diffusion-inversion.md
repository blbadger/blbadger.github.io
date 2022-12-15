## Diffusion Inversion Generative Models

An exploration of diffusion (aka denoising diffusion probabalistic) models. 

### Introduction with Denoising Autoencoders

Suppose one were to wonder how to de-noise an image that was corrupted some small amount.  One way to approach this problem is to specify what exactly noise looks like such that it may be filtered out from an input. But if the type of noise were not known beforehand, it would not be possible to identify exactly what the noisy versus original parts of the corrupted image were.  We therefore need some information on what the noise compared to the original image is, given samples of the original input $x$ and the corrupted samples $\widehat{x}$, it may be difficult to remove such noise in the general case.

As is often the case for poorly defined problems (in this case finding a general de-noiser), another way to address the problem of noise removal is to have a machine learning program learn how to accomplish noise removal in a probabalistic manner.  A trained program would ideally be able to remove each element of noise with high likelihood.

One of the most effective ways of removing noise probabalistically is to use a deep learning model as the denoiser.  Doing so allows us to refrain from specifying what the noise should look like, even in very general terms, which allows for a greater number of possible noise functions to be approximated by the model.  

### Introduction to Diffusion Inversion

Now suppose that instead of recovering an input that has been corrupted slightly, we want instead to recover an input that was severely corrupted and is nearly indistinguisheable from noise.  We could attempt to train our denoising autoencoder used above on samples with larger and larger amounts of noise, but doing so in most cases does not yield even a rough approximation of the desired input.  This is because we are asking the autoencoder to perform a very difficult task because most images that one would want to reconstruct are really nothing like a typical noise sample at all. 

To make this very difficult task approachable, we can break it into smaller sub-tasks that are more manageable.  This is the same approach taken to optimiziation of any deep learning algorithm: finding a minimal value via a direct method is intractable, so instead we use gradient descent and take many small steps towards a minimal point in the loss function space.  

### Using Diffusion to generate handwritten digits

Let's try to generate images of handwritten digits using diffusion inversion.  

### Attention augmented unet diffision
