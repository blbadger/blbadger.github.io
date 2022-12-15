## Diffusion Inversion Generative Models

An exploration of diffusion (aka denoising diffusion probabalistic) models. 

### Introduction with Denoising Autoencoders

Suppose one were to wonder how to de-noise an image that was corrupted some small amount.  One way to approach this problem is to specify what exactly noise looks like such that it may be filtered out from an input. But if the type of noise were not known beforehand, it would not be possible to identify exactly what the noisy versus original parts of the corrupted image were.  We therefore need some information on what the noise compared to the original image is, given samples of the original input $x$ and the corrupted samples $\widehat{x}$, it may be difficult to remove such noise in the general case.

As is often the case for poorly defined problems (in this case finding a general de-noiser), another way to address the problem of noise removal is to have a machine learning program learn how to accomplish noise removal in a probabalistic manner.  A trained program would ideally be able to remove each element of noise with high likelihood.

One of the most effective ways of removing noise probabalistically is to use a deep learning model as the denoiser.  Doing so allows us to refrain from specifying what the noise should look like, even in very general terms, which allows for a greater number of possible noise functions to be approximated by the model.  The loss function for a denoising autoencoder is as follows:

$$
L = || x - g(f(\widehat{x})) ||_2 = || x - g(h) ||
$$

where $f(x)$ signifies the encoder which maps corrupted input $\widehat{x}$ to the hidden state $h$, also known as the 'code', and $g(h)$ maps this hidden state to the output.

### Introduction to Diffusion Inversion

Now suppose that instead of recovering an input that has been corrupted slightly, we want instead to recover an input that was severely corrupted and is nearly indistinguisheable from noise.  We could attempt to train our denoising autoencoder used above on samples with larger and larger amounts of noise, but doing so in most cases does not yield even a rough approximation of the desired input.  This is because we are asking the autoencoder to perform a very difficult task because most images that one would want to reconstruct are really nothing like a typical noise sample at all. 

To make this very difficult task approachable, we can break it into smaller sub-tasks that are more manageable.  This is the same approach taken to optimiziation of any deep learning algorithm: finding a minimal value via a direct method is intractable, so instead we use gradient descent and take many small steps towards a minimal point in the loss function space.  

In this context, we will employ an autoencoder to learn how to take very small steps to de-noise an input (assuming Gaussian noise), and then generate images by reversing this process with the starting point of pure noise.  Doing so is equivalent to diffusion inversion, also called denoising diffusion probabalistic modeling or simply diffusion models for short.


### Using Diffusion to generate handwritten digits

Let's try to generate images of handwritten digits using diffusion inversion.  First we need an autoencoder, and for that we can turn to a miniaturized and fully connected version of the well-known [U-net](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) architecture introduced by Ronnenberger and colleagues.  The U-net architecture will be considered in more detail later, as for now we will make do with the fully connected version shown below:

![fully connected unet architecture]({{https://blbadger.github.io}}/neural_networks/fc_unet.png)

This is a variation of a typical undercomplete autoencoder which compresses the input somewhat (2k) using fully connected layers followed by de-compression where residual connections are added between layers of equal size, forming a distinctive U-shape.  It should be noted that layer normalization may be ommitted without any noticeable effects.  Layer sizes were determined upon some experimentation, where it was found that smaller (ie starting with 1k) widths trained very slowly. 

Next we chose to take 1000 steps in the forward diffusion process, meaning that our model will learn to de-noise an input over 1k small steps in the diffusion inversion generative process as well.  It is now necessary (for learning in a reasonable amount of time) to provide the time-step information to the model in some way: this allows the model to 'know' what time-step we are at and therefore approximately how much noise exists in the input.  [Ho and colleagues](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html) use positional encoding (inspired by the original transformer positional encodings for sequences) and therefore encode the time-step information as a trigonometric function added to the input (after a one-layer feedforward trainable layer).  Although this approach would certainly be expected to be effective for a small model as well as large ones, we take the simpler approach of encoding time-step information in a single input element before appending that element to the others after flattening.  Specifically, this time-step element $x_t = t / T$ where $t$ is the current time-step and $T$ is the final time-step.

An aside: it is interesting that time information is required for diffusion inversion at all, being that one might expect for an autoencoder trained to estimate the variance of the input noise relative to some slightly less-noisy input to be capable of estimating the accumulated noise as well, and therefore estimate the time-step value for itself.  Removing the time-step information from the diffusion process of either the original U-net or the small autoencoder above yields poor sample generation but curiously the models are capable of optimizing the objective function without much trouble.  It is currently unclear why time-step information is required for sample generation, ie $x_{T} \to x_{0}$, but not for minimizing $\epsilon - \epsilon(\sqrt \alpha_t x_0 + \sqrt{1 - \alpha_t}\epsilon). 

Concatenation of a single element time-step value to the input tensor may be done as follows,

```python
class FCEncoder(nn.Module):

	def __init__(self, starting_size, channels):
		super().__init__()
		starting = starting_size
		self.input_transform = nn.Linear(32*32*3 + 1, starting)
    ...

	def forward(self, input_tensor, time):
		time_tensor = torch.tensor(time/timesteps).reshape(batch_size, 1)
		input_tensor = torch.flatten(input_tensor, start_dim=1)
		input_tensor = torch.cat((input_tensor, time_tensor), dim=-1)
    ...
```

and the fully connected autoencoder combined with this one-element time information approach yields decent results after training on non-normalized MNIST data as shown below.

![mnist diffusion]({{https://blbadger.github.io}}/neural_networks/mnist_diffusion.png)

These results are somewhat more impressive when we consider that generative models typically struggle somewhat with un-normalized MNIST data, as most elements in the input are identically zero.  Experimenting with different learning rates and input modifications convinces us that diffusion inversion is happily rather insensitive to exact configurations, which is a marked difference from GAN training. 

### Attention augmented unet diffision

For higher-resolution images the use of unmodified fully connected architectures is typically infeasible due to the very large number of parameters resulting.

![lsun churches 64 diffusion]({{https://blbadger.github.io}}/neural_networks/diffusion_cover.png)
