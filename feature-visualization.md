## Feature Visualization and Deep Dreams

### Introduction

Suppose one wished to perform a very difficult task, but did not know how to begin.  In mathematical problem solving, one of the strategies most commonly used to approach a seemingly impossible problem is to convert that problem into an easier one, and solve that.  The principle 'make it easier' is considered by some mathematical competition instructors (Zeitz in particular) as being perhaps the most important of all problems.

In many respects, the field of deep learning places the same importance on the 'make it easier' principle.  Most problems faced by the field certainly fall into the category of being very difficult: object recognition, indeed, is so difficult that no one has made much progress in this problem by directly programming a computer with explicit instructions in decades.  The mathematical function mapping an input image to its corresponding output classification is therefore very difficult to find indeed.

Classical machine learning uses a combination of ideas from statistics, information theory, and computer science to try to find this and other difficult functions in a limited number of steps (usually one). Deep learning distinguishes itself by attempting to approximate functions using more than one step, such that the input is represented in a better and better way before a final classification is made.  In neural networks and similar deep learning approaches, this representation occurs in layers of nodes (usually called neurons).  

As we will be employing a model whose hidden layers consists primarily of 2-dimensional convolutions, first it may be beneficial to orient ourselves to the layout of a standard example of this type of layer.  See [this page](https://blbadger.github.io/neural-networks.html#convolutions-explained) for a recap of what a convolution is and why this operation is so useful for deep learning models dealing with images.

For a convolutional layer with a single kernal, the output is a two-dimensional grid corresponding to the value of the convolutional operation centered on each of the pixels.  Therefore it could be expected that an image of 256 pixels by 256 pixels, ie a 256x256 tensor with each pixel intensity value corresponding to one tensor element, when passed through a convolutional layer will give an output tensor of 256 by 256 (if a specific type of padding is used to deal with edge pixels).  There are normally multiple kernals used per convolutional layer, so for a layer with say 16 kernals (also called 'filters') we would have a 16x256x256 output.  Each of the 16 elements are referred to as a 'feature map', and on this page will always precede the other tensor elements.

### Feature Visualization from First Principles

The process of sequential input representation, therefore, is perhaps the defining feature of deep learning.  This representation does make for one significant difficulty, however: how does one know how the problem was solved?  We can understand how the last representation was used to make the output in a similar way to how other machine learning techniques give an output from an input, but what about how the actual input is used?  The representations may differ significantly from the input itself, making the general answer to this question very difficult.

One way to address this question is to try to figure out which elements of the input contribute more or less to the output.  This is called input attribution.

Another way is to try to understand what the representation of each layer actually respond to with regards to the input, and then try to understand how this response occurs.  Perhaps the most natural way to study this is to feed many input into the model and observe which input give a higher activation to any layer or neuron or even subset of neurons across many layers, and which images yield a lower activation.  


There is a problem with this approach, however: images as a rule do not contain only one characteristic that could become a feature, and furthermore even if some did there is little way of knowing how features would be combined together between layers being that nonlinear transformations are applied to all layers in a typical deep learning model.  

Instead we can approach the question of what each neuron, subset of neurons, or layer responds to by having the neurons in question act as if they are our model's outputs, and performing gradient ascent on the input.  We have seen [elsewhere](https://github.com/blbadger/blbadger.github.io/blob/master/input-generation.md) that an output can effectively specify the expected image on the input using gradient ascent combined with a limited number of priors that are common to nearly all natural images (pixel correlation, transformational resiliency, etc.) so it is logical to postulate that the same may be true for outputs that are actually hidden layers, albeit that maximizing the activation of a hidden layer neuron or layer does not have an intrinsic meaning w.r.t the true output.

To make this more precise, what we are searching for is an input $a'$ such that the activation of our chosen hidden layer or neuron $z^l$ (given the model configuration $\theta$ and input $a$) denoted by $z^l(a, \theta)$ is maximized,

$$
a' = \underset{a} \mathrm{arg \; max} \; z^l(a, \theta)
$$

For a two-dimensional convolutional layer with $m$ rows and $n$ columns, the total activation at layer (ie feature) $l$ is

$$
z^l = \sum_m \sum_n z^l_{m, n}
$$

In tensor notation, this would be written as `[zl, :, :]` where `:` indicates all elements of the appropriate index.

For a subset of neurons in layer $l$, say all neruons in row $n$, 

$$
z^l_m = \sum_n z^l_{m, n}
$$

which is denoted `[zl, m, :]`, and that the element of row `m` and column `n` is denoted `[xl, m, n]`.

Finding the exact value of $a'$ can be very difficult for non-convex functions like hidden layer outputs. An approximation for the input $a'$ such that when given to our model gives an approximate maximum value of $z^l(a', \theta)$ may be found via gradient descent.  The gradient in question used on this page is the gradient of the $L_1$ metric between large constant $C$ and the activation of a specific layer (or a subset of this layer) $z^l$

$$
g = \nabla_a (C - z^l(a, \theta))
$$

At each step of the gradient descent procedure, the input at point $a_k$ is updated to make $a_{k+1}$ as follows

$$
a_{k+1} = a_k - \epsilon * g
$$

with $*$ usually denoting broadcasted multiplication from scalar $\epsilon$ to tensor $g$.  The hope is that the sequence $a_k, a_{k+1}, a_{k+2}, ..., a_j$ converges to $a'$, which happily is normally the case for appropriately chosen values of $\epsilon$ for most layers.

We will be using Pytorch for our experiments on this page.  As explained [elsewhere](https://blbadger.github.io/input-generation.html#input-generation-with-auxiliary-outputs), Pytorch uses symbol-to-number differentiation as opposed to symbol-to-symbol differentiation which means that the automatic differentiation engine must be told which gradients are to be computed before forward propegation begins. Practically speaking, this means that getting the gradient of a model's parameters with respect to a hidden layer is difficult without special methods. 

Rather than deal with these somewhat tricky special functions, we can instead simply modify the model in question in order to make whichever layer is of interest to be our new output.  This section will focus on the InceptionV3 model, which is as follows:

![inceptionv3 architecture]({{https://blbadger.github.io}}/neural_networks/inception3_labelled.png)

(image retrieved and modified from Google [docs](https://cloud.google.com/tpu/docs/inception-v3-advanced))

and may be found on the following [github repo](https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py).  Inspecting this repository will show us how the inceptionv3 model has been implemented, and that there are a few differences between the origina model and the open-source Pytorch version (most notably that the last layers do not contain softmax layers by default).

A pre-trained version of InceptionV3 may be loaded from `torch.hub` 

```python
Inception3 = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True).to(device)
```
And after deciding on which layer is of interest, the model may be modified by writing a class that takes the original InceptionV3 model as an initialization argument and then specifies what to do with the layers in question.  As this is a pre-trained model, it generally does not make sense to change the configuration of the layers themselves but we can modify the model to output any layer we wish.  Here we want to view the activations of the`Conv2d_3b_1x1` layer

```python
class NewModel(nn.Module):

	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, x):
		# N x 3 x 299 x 299
		x = self.model.Conv2d_1a_3x3(x)
		# N x 32 x 149 x 149
		x = self.model.Conv2d_2a_3x3(x)
		# N x 32 x 147 x 147
		x = self.model.Conv2d_2b_3x3(x)
		# N x 64 x 147 x 147
		x = self.model.maxpool1(x)
		# N x 64 x 73 x 73
		x = self.model.Conv2d_3b_1x1(x)
		return x
```
and because we are taking the original trained model as a parameter, each layer maintains its trained weight and bias.

Instantiating this model with the original InceptionV3 model object

```python
newmodel = NewModel(Inception3)
```
we can now use our `newmodel` class to find the activation of the `Conv2d_3b_1x1` layer and perform backpropegation to the input, which is necessary to find the gradient of the output of this layer with respect to the input image.

To be specific, we want to find the gradient of the output of our layer in question $O^l$ with respect to the input $x$

$$
g = \nabla_a O^l(a, \theta)
$$

```python
def layer_gradient(model, input_tensor, desired_output):
  ...
	input_tensor.requires_grad = True
	output = model(input_tensor).to(device)
	focus = output[0][200][:][:] # the target to maximize the output
	target = torch.ones(focus.shape).to(device)*200 # make a large target of the correct dims
	loss = torch.sum(target - focus)
	loss.backward()
	gradient = input_tensor.grad
	return gradient
```

Now that we have the gradient of the output (which is our hidden layer of interest in the original InceptionV3 model) with respect to the input, we can perform gradient descent on the input in order to maximize the activations of that layer.  Note that this process is sometimes called 'gradient ascent', as we are modifying the input in order to maximize a value in the output.  But as our loss is the $L_1$ distance from a tensor with all elements assigned to be a large constant $C=200$, gradient descent is the preferred terminology on this page as this loss is indeed minimized by moving against the gradient.

For [this page](https://blbadger.github.io/input-generation.html) These restrictions may be thought of as Bayesian priors that we introduce, knowing certain characteristics that natural images possess.  The three characteristics that are implemented below are relative smoothness (enforced by Guassian convolution via `torchvision.transforms.functional.gaussian_blur`) and transformational invariance (using `torchvision.transforms.Resize()`).

We can use this procedure of gradient descent on the input combined with priors to visualize the input generated by maximizing the output of the given neuron, or layer.  If we choose module `mixed_6b` and maximize the activations of neurons in the 654th (of 738 possible) feature map of this module, we find an interesting pattern is formed.

![654 visualization]({{https://blbadger.github.io}}/neural_networks/inception_654.png)

The first thing to note is that the tensor indicies for 'row' and 'column' do indeed accurately reflect the position of the pixels affected by maximization of the output of the neuron of a given position.

It is interesting to consider how exactly the feature pattern forms from tiles of each row and column's contribution.  For each feature map in module `mixed_6b`, we have a 17x17 grid and can therefore iterate through each element of the grid, left to right and top to bottom, like reading a book.  Slicing tensors into non-rectangular subsets can be difficult, so instead the process is to compose the loss of two rectangular slices: one for all columns of each row preceding the current row, and another rectangle for the current row up to the appropriate column.

```python
def layer_gradient(model, input_tensor, desired_output, index):
	...
	input_tensor.requires_grad = True
	output = model(input_tensor).to(device)
	row, col = index // 17, index % 17
	if row > 0:
		focus = output[0, 415, :row, :] # first rectangle of prior rows
		focus2 = output[0, 415, row, :col+1] # second rectangle of the last row
		target = torch.ones(focus.shape).to(device)*200
		target2 = torch.ones(focus2.shape).to(device)*200
		loss = torch.sum(target - focus)
		loss += torch.sum(target2 - focus2) # add losses

	else:
		focus = output[0, 415, 0, :col+1]
		target = torch.ones(focus.shape).to(device)*200
		loss = torch.sum(target - focus)

	loss.backward()
	gradient = input_tensor.grad
	return gradient
```

Note that we must also set our random seed in order to make reproducable images.  This can be done as follows:

```python
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
```

and this results in

{% include youtube.html id='EJo1fUzheSU' %}

We come to an intersting observation: each neuron when optimized visibly affects only a small area that corresponds to that neuron's position in the `[row, col]` position in the feature tensor, but when we optimize multiple neurons the addition of one more affects the entire image, rather than the area that it would affect if it were optimized alone.  This can be seen by observing how the rope-like segments near the top left corner continue to change upon addition of neurons that alone only affect the bottom-right corner.

How is this possible? Neurons of any one particular layer are usually considered to be linearly independent units, and assumption that provides the basis for being able to apply the gradient of the loss to each element of a layer during training.  But if the neurons of layer `mixed_65 654` were linearly independent with respect to the input maps formed when optimizing these neuron's activations, we would not observe any change in areas unaffected by each neuron.  It is at present not clear why this pattern occurs.

For the 415th feature map of the same module, a somewhat more abstract pattern is formed.  Observe how a single neuron exhibits a much broader field of influence on the input when it is optimized compared to the previous layer: this is a common feature of interior (ie not near layer 0 or 738) versus exterior (close to 0 or 748 in this particular module).  This is likely the result of differences in the inception module's architecture for different layers that have been concatenated to make the single `mixed_6b` output.

![654 visualization]({{https://blbadger.github.io}}/neural_networks/inception_415.png)

Something interesting to note is that the neuron at position $[415, 9, 9]$ in the `mixed_6b` module is activated by a number of different patterns at once, which is notably different than the neuron at position $[354, 6, 0]$ which is maximally activated by one rope-like pattern alone.


The above method for introducing transformational invariance leads to fairly low-resolution images by design (for speed). 
The Octave method without cropping or padding may be implemented as follows:

```python
def octave(single_input, target_output, iterations, learning_rates, sigmas):
	...
	start_lr, end_lr = learning_rates
	start_sigma, end_sigma = sigmas

	for i in range(iterations):
		single_input = single_input.detach() # remove the gradient for the input (if present)
		input_grad = layer_gradient(Inception, single_input, target_output) # compute input gradient
		single_input -= (start_lr*(iterations-i)/iterations + end_lr*i/iterations) * input_grad # gradient descent step
		single_input = torchvision.transforms.functional.gaussian_blur(single_input, 3, sigma=(start_sigma*(iterations-i)/iterations + end_sigma*i/iterations)) # gaussian convolution

	return single_input
```

This function may now be called to perform gradient descent with scaling, and in the following method three octaves are applied: one without upscaling, one that scales to 340 pixels, and one that scales to 380.

```python
def generate_singleinput(model, input_tensors, output_tensors, index, count, random_input=True):
	...
	single_input = octave(single_input, target_output, 100, [0.4, 0.3], [2.4, 0.8])
 
	single_input = torchvision.transforms.Resize([340, 340])(single_input)
	single_input = octave(single_input, target_output, 100, [0.3, 0.2], [1.5, 0.4])

	single_input = torchvision.transforms.Resize([380, 380])(single_input)
	single_input = octave(single_input, target_output, 100, [0.2, 0.1], [1.1, 0.3])
```

The images produced are of higher resolution and are generally clearer than with only down-sampling used, so most of the rest of the images on this page are generated via gradient descent with octaves.

It is worth assessing how representative the images produced using octaves and Gaussian convolutions with gradient descent are to the images produced with gradient descent alone.  If the resulting images are completely different, there would be limited utility in using octaves to understand how features recognize an input.  For four arbitrarily chosen features of module Mixed 6b, plotting pure gradient descent compared to octave-based gradient descent is as follows:

![inceptionv3 comparison]({{https://blbadger.github.io}}/neural_networks/Inception3_raw_octave_comparison.png)

The images are clearly somewhat different, but the qualitative patterns are more or less unchanged. These results have been observed for features in other layers as well, providing the impetus for proceeding with using octaves and Gaussian convolutions with gradient descent to optimize the layers in images on the rest of the page.

### Mapping InceptionV3 Features

In the pioneering work on feature visualization in the original Googlenet architecture (aka InceptionV1), [Olah and colleagues](https://distill.pub/2017/feature-visualization/) observed an increase in the complexity of the images resulting from maximizing the activation of successive layers in that model: the deeper into the network the authors looked, the more information could be gleaned from the input after performing gradient descent to maximize the layer's activation.  Early layers show maze-like patterns typical of early convolutional layers, which give way to more complicated patterns and textures and eventually whole objects.  Curiously, the last layers were reported to contain multiple objects jumbled together, a phenomenon this author has observed [elsewhere](https://blbadger.github.io/input-generation.html).

The InceptionV3 architecture differs somewhat from the GoogleNet that was thoroughly explored, so it is worth investigating whether or not the same phenomenon is observed for this model as well.  To orient ourselves, the module map for the InceptionV3 model may be divided into four general regions.

![inceptionv3 architecture]({{https://blbadger.github.io}}/neural_networks/inception3_zones.png)

In the first of these regions, convolutions are applied in a sequential manner with no particularly special additions.  Optimizing the first 16 features of each layer, we have

![inceptionv3 architecture]({{https://blbadger.github.io}}/neural_networks/inception3_entry_convs.png)

It is interesting to note that there are few in the input when we optimize for activation of the Conv2d 1a layer. Instead, it appears that the InceptionV3 network has learned a modified version of the CMYK color pallate, in which yellow has been substituted for a combination of yellow and blue (ie green). To this author's knowledge it has not been found previously that a deep learning model has learned a color pallate, so it is unknown whether or not this behavior is specific to Inceptionv3.  It should be noted that if optimization occurs without the octave process, similar colors are observed but a fine grid-like pattern reminiscent of CRT video appears for a minority of features.

Note that the gray filters signify a lack of gradient in that feature with respect to the input. 

The second convolutional layer begins to respont to patterns in the image, and in the following layers the features become more detailed.

Next we come to the mixed convolutional layers, which are composed of modules (sometimes called inception units) that contain convolutional layers in parallel, which become concatenated together at the end of the module to make one stack of the features combining all features from these units.  The first three mixed units

![inceptionv3 architecture]({{https://blbadger.github.io}}/neural_networks/Inception3_early_mixed.png)

At the middle mixed layers, we have

![inceptionv3 architecture]({{https://blbadger.github.io}}/neural_networks/Inception3_middle_mixed.png)

and in the late mixed layers, many patterns appear 

![inceptionv3 architecture]({{https://blbadger.github.io}}/neural_networks/Inception3_late_mixed.png)

### Layer and Neuron Interactions














