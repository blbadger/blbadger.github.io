## Feature Visualization and Deep Dreams

### Introduction

Suppose one wished to perform a very difficult task, but did not know how to begin.  In mathematical problem solving, one of the strategies most commonly used to approach a seemingly impossible problem is to convert that problem into an easier one, and solve that.  The principle 'make it easier' is considered by some mathematical competition instructors (Zeitz etc.) as being perhaps the most important of all problems.

In many respects, the field of deep learning places the same importance on the 'make it easier' principle.  Most problems faced by the field certainly fall into the category of being very difficult: object recognition, indeed, is so difficult that no one has made much progress in this problem by directly programming a computer with explicit instructions in decades.  The mathematical function mapping an input image to its corresponding output classification is therefore very difficult to find indeed.

Classical machine learning uses a combination of ideas from statistics, information theory, and computer science to try to find this and other difficult functions in a limited number of steps (usually one). Deep learning distinguishes itself by attempting to approximate functions using more than one step, such that the input is represented in a better and better way before a final classification is made.  In neural networks and similar deep learning approaches, this representation occurs in layers of nodes (usually called neurons).  

As we will be employing a model whose hidden layers consists primarily of 2-dimensional convolutions, first it may be beneficial to orient ourselves to the layout of a standard example of this type of layer.  See [this page](https://blbadger.github.io/neural-networks.html#convolutions-explained) for a recap of what a convolution is and why this operation is so useful for deep learning models dealing with images.

For a convolutional layer with a single kernal, the output is a two-dimensional grid corresponding to the value of the convolutional operation centered on each of the pixels.  Therefore it could be expected that an image of 256 pixels by 256 pixels, ie a 256x256 tensor with each pixel intensity value corresponding to one tensor element, when passed through a convolutional layer will give an output tensor of 256 by 256 (if a specific type of padding is used to deal with edge pixels).  There are normally multiple kernals used per convolutional layer, so for a layer with say 16 kernals (also called 'filters') we would have a 16x256x256 output.  Each of the 16 elements are referred to as a 'feature map', and on this page will always precede the other tensor elements.

### Feature Visualization 

The process of sequential input representation, therefore, is perhaps the defining feature of deep learning.  This representation does make for one significant difficulty, however: how does one know how the problem was solved?  We can understand how the last representation was used to make the output in a similar way to how other machine learning techniques give an output from an input, but what about how the actual input is used?  The representations may differ significantly from the input itself, making the general answer to this question very difficult.

One way to address this question is to try to figure out which elements of the input contribute more or less to the output.  This is called input attribution.

Another way is to try to understand what the representation of each layer actually respond to with regards to the input, and then try to understand how this response occurs.  Perhaps the most natural way to study this is to feed many input into the model and observe which input give a higher activation to any layer or neuron or even subset of neurons across many layers, and which images yield a lower activation.  

For a convolutional layer with $n$ filters, the total activation (pre-nonlinearity applied) at layer $l$ would be 

$$
z^l = \sum_m \sum_n z^l_{m, n}
$$

There is a problem with this approach, however: images as a rule do not contain only one characteristic that could become a feature, and furthermore even if some did there is little way of knowing how features would be combined together between layers being that nonlinear transformations are applied to all layers in a typical deep learning model.  

Instead we can approach the question of what each neuron, subset of neurons, or layer responds to by having the neurons in question act as if they are our model's outputs, and performing gradient ascent on the input.  We have seen [elsewhere](https://github.com/blbadger/blbadger.github.io/blob/master/input-generation.md) that an output can effectively specify the expected image on the input using gradient ascent combined with a limited number of priors that are common to nearly all natural images (pixel correlation, transformational resiliency, etc.) so it is logical to postulate that the same may be true for outputs that are actually hidden layers, albeit that maximizing the activation of a hidden layer neuron or layer does not have an intrinsic meaning w.r.t the true output.

We will be using Pytorch for our experiments on this page.  As explained [elsewhere](https://blbadger.github.io/input-generation.html#input-generation-with-auxiliary-outputs), Pytorch uses symbol-to-number differentiation as opposed to symbol-to-symbol differentiation which means that the automatic differentiation engine must be told which gradients are to be computed before forward propegation begins. Practically speaking, this means that getting the gradient of a model's parameters with respect to a hidden layer is difficult without special methods. 

Rather than deal with these somewhat tricky functions, we can instead simply modify the model in question in order to make whichever layer is of interest to be our new output.  This section will focus on the InceptionV3 model, which is as follows:

![inceptionv3 architecture]({{https://blbadger.github.io}}/neural_networks/inceptionv3_architecture.png)

(image source: Google [docs](https://cloud.google.com/tpu/docs/inception-v3-advanced))

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

To be specific, we want to find the gradient of the output of our layer in question $O(l)$ with respect to the input $x$

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
	loss = torch.sum(target - focus) # + torch.sum(target2 - focus2)
	loss.backward()
	gradient = input_tensor.grad
	return gradient
```

We can use this procedure to visualize the input generated by maximizing the output of the given neuron, or layer.  If we choose module `mixed_6b` and maximize the activations of neurons in the 654th (of 738 possible) feature map of this module, we find an interesting pattern is formed.

![654 visualization]({{https://blbadger.github.io}}/neural_networks/inception_654.png)

When the same procedure is applied to the 415th feature map of the same module, a somewhat more abstract pattern is formed.

![654 visualization]({{https://blbadger.github.io}}/neural_networks/inception_415.png)

















