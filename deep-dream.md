## Feature Visualization II: Deep Dream

### Introduction: Feature Optimization

In [Part I](https://blbadger.github.io/feature-visualization.html), deep learning model convolutional features were investigated by constructing an input that resulted in the highest total activation in that feature, starting with random noise.  But the training dataset for the models used (ImageNet) does not contain any images that resemble pure noise, making the initial input $a_0$ somewhat artificial.  What happens if we start with a real image for $a_0$ rather than noise and modify that image using gradient descent?

To recap, an input $a$ may be modified via gradient descent where the gradient at step $n$, denoted $g_n$, is some loss function of the output $J(O)$ on the input $a_n$, given model parameters $\theta$,

$$
g_n = \nabla_a J(O(a_n, \theta))
$$

On the page referenced above, we optimized the output activation of interest $\widehat y_n$ by assigning the loss function of the output layer $J(O)$ to be the difference  $J(O(a, \theta)) = C - \widehat y_n$ where $C$ is some large constant and the initial input $a_0$ is a scaled normal distribution.  But gradient descent alone was not found to be very effective in producing recognizable images on $a_0$, so two additional Bayesian priors were added: smoothness (ie pixel cross-correlation) via Gaussian convolution $\mathcal{N}$ and translational invariance with Octave-based jitter, here denoted $\mathscr{J}$, were employed with the gradient for input $a_n$, denoted $g_n$.  The actual update procedure is

$$
a_{n+1} =\mathscr{J} \left( \mathcal{N}(a_n + \epsilon g_n) \right)
$$

Now we will investigate a similar update, but using octaves without jitter (denoted $\mathscr{O}$).  This is done by increasing the image resolution and feeding the entire image into the model, rather than a randomly cropped version.  

$$
a_{n+1} =\mathscr{O} \left( \mathcal{N}(a_n + \epsilon g_n) \right)
$$

A three-octave approach was found to result in clear images.

```python
# original image size: 299x299
single_input = directed_octave(single_input, target_output, 100, [0.5, 0.4], [2.4, 0.8], 0, index, crop=False)

single_input = torchvision.transforms.Resize([380, 380])(single_input)
single_input = directed_octave(single_input, target_output, 100, [0.4, 0.3], [1.5, 0.4], 330, index, crop=False)

single_input = torchvision.transforms.Resize([460, 460])(single_input)
single_input = directed_octave(single_input, target_output, 100, [0.3, 0.2], [1.1, 0.3], 375, index, crop=False)

```

We will start by optimizing one or two features at a time.  To do this, we want to increase the activation of each neuron in that feature using gradient descent. One way to accomplish this is to assign the loss function $J(O(a, \theta))$ to be the sum of the difference between each neuron's activation in the feature of interest, as a tensor denoted $z^l_f$, and some large constant $C$.

$$
g = \nabla_a(J(O(a; \theta))) = \nabla_a(C - z^l_f) \\
= \nabla_a \left(\sum_m \sum_n C - z^l_{f, m, n} \right) \\
$$

As was done for feature visualization, computation of the gradient can be implemented with $C=200$ as follows:

```python
def layer_gradient(model, input_tensor, desired_output, index):
	...
	input_tensor.requires_grad = True
	output = model(input_tensor).to(device)
	focus = output[0][:][:][:] # optimize the entire layer
	target = torch.ones(focus.shape).to(device)*200
	loss = torch.sum(target - focus)
	loss.backward()
	gradient = input_tensor.grad
	return gradient
```

For this procedure applied to 8 features from layer Mixed 6d in InceptionV3 $a_0$ are selections of flowers, we have

![Dream]({{https://blbadger.github.io}}/neural_networks/InceptionV3_mixed6d_dream.png)

One can also modify the input $a$ such that multiple layers are maximally activated: here we have features from layers 'Mixed 6d' and 'Mixed 6e' jointly optimized by assigning the total loss to be the sum of the loss of each feature, again using a collection of flowers $a_0$.

![Dream]({{https://blbadger.github.io}}/neural_networks/InceptionV3_mixed6d_Mixed6e_dream.png)

### Layer Optimization

It may be wondered what would happen if the input image were optimized without enforcing a prior requiring smoothness in the final image.  Omitting smoothness has a fairly clear justification in this scenario: the starting input is a natural image that contains the smoothness and other statistical features of other natural images, so as long as we do not modify this image too much then we would expect to end with something that resembles a natural image.

This being the case, we can implement gradient descent using octaves but without Gaussian convolution as follows:

```python

def octave(single_input, target_output, iterations, learning_rates, sigmas, index):
	...
	start_lr, end_lr = learning_rates
	start_sigma, end_sigma = sigmas
	for i in range(iterations):
		single_input = single_input.detach() # remove the gradient for the input (if present)
		input_grad = layer_gradient(newmodel, input, target_output, index) # compute input gradient
		single_input -= (start_lr*(iterations-i)/iterations + end_lr*i/iterations)* input_grad # gradient descent step
	return single_input
```

The other change we will make is to optimize the activations of an entire layer rather than only one or two features.  

$$
z^l = \sum_f \sum_m \sum_n z^l_{f, m, n}
$$

which is denoted as the sum of the tensor `[:, :, :]` of layer $l$. Using InceptionV3 as our model and applying gradient descent in 3 octaves to an input of flowers, when optimizing layer Mixed 6b we have

![Dream]({{https://blbadger.github.io}}/neural_networks/dream_mixed6b.png)

Other models yield similar results, for example optimizing layer 3 of ResNet gives

![Dream]({{https://blbadger.github.io}}/neural_networks/Resnet50_layer3_dream.png)

### Image Resolution Flexibility

One interesting attribute of convolutional layers is thay they may be applied to inputs of arbitrary dimension.

As for feature maps, we can apply the gradient descent procedure on inputs of abitrary resolution because each layer is convolutional, meaning that only the weights and biases of kernals are specified such that any size of image may be used as input $a$ without having to change the model parameters $\theta$.  Note that this is only true if all layers up to the final are convolutional: as soon as one desires use of a fully connected architecture, it is usually necessary to use a fixed input size. Right click for images at full resolution.

![Dream]({{https://blbadger.github.io}}/neural_networks/Inception3_dream_mixed6b_layer.png)

![Dream]({{https://blbadger.github.io}}/neural_networks/Inception3_dream_mixed6b_layer_hres2.png)

Note, however, that there is a clear limit to this procedure: although the input image resolution may be increased without bound, the convolutional kernals of the model in question (here InceptionV3) during training were learned only for the resolution that training inputs existed at.  This is why an increase in resolution yields smaller details introduced relative to the whole image, as the changes during gradient descent are made by a model that learned to expect images of a lower resolution.

### How Deep Dream makes Coherent Images

It may be wondered how deep dream makes any kind of recognizable image at all.  For [input image generation](https://blbadger.github.io/input-generation.html), we saw that gradient descent on an initial input of random noise did not yield any recognizable images, and this was for only one neuron or feature at a time rather than for many features as we have here.  It was only after the addition of a smoothness Bayesian prior that gradient descent was able to begin to produce recognizable images, but smoothness is typically not added during deep dream.

Futhermore, when one considers how a convolutional layer works for image classification, it is not immediately clear how optimizing the activation of many layers together would give anything other than an incoherent jumble.  This is because during feed-forward operation each feature map in a convolutional layer is expected to have a different activation corresponding to which features are found in the image one wants to classify.  Activating all the feature maps is equivalent to saying that one wants an image that has all possible features, and for deeper layers with many (>= 256) features one may reasonably expect that this image will not look like much of anything at all, especially when no smoothness constraint is enforced.

After some consideration, the situation may not seem as bad as at first glance.  The deep dream process begins using a natural image rather than noise, and therefore although we don't enforce statistical characteristics of natural images during gradient descent we have enforced them at the beginning of gradient descent.  

To see why optimizing the activation of many features at once does not necessarily lead to a nonsensical image, consider the following example in which first one, then five, and then all 768 features of InceptionV3's layer Mixed 6c are optimized.  As we saw in [part 1](https://blbadger.github.io/feature-visualization.html), the following maps were found to maximally activate each of the first five features of layer Mixed6c:

![deep dream explanation]({{https://blbadger.github.io}}/neural_networks/deep_dream_explanation2.png)

These inputs were generated from noise, but nevertheless if one particular feature were to be activated by a certain pattern, one would expect this to be the case if the starting point were not noise but instead an image of some kind.  Indeed we see that is the case: observe how feature 4 gives a very similar yarn-like appearance when the starting point is a picture of a dalmation (top right below) as when it is noise (above).

![deep dream explanation]({{https://blbadger.github.io}}/neural_networks/deep_dream_explanation.png)

On the bottom row, multiple features are optimized simultaneously.  It is necessary to scale back the gradient to avoid producing very high-frequency or saturated final inputs, and we can do this by simply weighting the gradient of the entire layer by a fraction corresponding to the inverse of the number of features in that layer: ie if there are around 1000 features in a given layer, we can divide the gradient of the layer by 1000. This is because the gradient is a linear operator and is therefore additive, meaning that the gradient of an entire layer $z^l$ is equivalent to the gradient of each feature added together,

$$
g = \nabla_a(J(O(a; \theta))) = \nabla_a(z^l) \\
= \nabla_a \left( \sum_f \sum_m \sum_n z^l_{f, m, n} \right) \\
= \sum_f \nabla_a \left( \sum_m \sum_n z^l_{f, m, n} \right)
$$

Therefore the gradient descent update performed may be scaled by the constant $b$ while keeping the same update $\epsilon$ as was used for optimization for an individual feature.  In the example above, $b=1/1000$ and

$$
a_{n+1} = a_n - \epsilon * b * g_n
$$

How then can we hope to make a coherent image if we are adding small gradients from nearly 1000 features?  The important thing to remember is that a single image cannot possibly optimize all features at once.  Consider the following simple example where we want to optimize the output of two features, where the gradient of the first feature $g_0$ for a 2x2 input $a$ is

$$
g_0 = 
\begin{bmatrix}
1 & -1  \\
-1 & 1  \\
\end{bmatrix}
$$

whereas another feature's gradient $g_1$ is

$$
g_2 = 
\begin{bmatrix}
-1 & 1  \\
1 & -1  \\
\end{bmatrix}
$$

now as gradients are additive, the total gradient is

$$
g = 
\begin{bmatrix}
0 & 0  \\
0 & 0  \\
\end{bmatrix}
$$

which when applied to the original input $a$ will simply yield $a$, so clearly neither feature's activations are optimized.

What this means is that the deep dream procedure gives what can be thought of as a kind of 'average' across all features in the layers of interest.  As each layer must pass the entirety of the necessary information from the input to the output for accurate classification, one can expect for the dream procedure to produce shapes that are most common in the training dataset used, provided the features recognizing these images may be optimized from the input.  This is why deep dream performed on sufficiently deep layers generally introduces animal objects, whereas a dataset trained on landscapes generates images of landscapes and buildings during deep dream ([reference](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)).

We can observe this inability to optimize all features during the dream process. Observe how the average feature activation does not change for many features even after 3 octaves of deep dream:

{% include youtube.html id='l__HrW5spn0' %}

### Enhancement with Layer Optimization

In the previous section it was observed that not all features of the layer of interest were capable of increasing their mean activation during deep dream.  Do certain features tend to exhibit an increase in activation, leading to a similar change in the input image for all starting images or does the image generated (and therefore the features that are activated) depend on the original input?  In other words, if the dream procedure tends to generate an image that is representative of the more common examples in the dataset used to train a network, what influence does the original input image have?  In the original deep dream [article](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html), it was observed that the dream procedure tends to amplify objects that the model's layer of interest 'sees' in original images.  It remains to be seen what influence the input has on a dream, or whether the same objects tend to be amplified no matter what input is given.

We will use GoogleNet as the model for this section. For reference, this model's architecture is as follows:

![annotated googlenet]({{https://blbadger.github.io}}/neural_networks/annotated_googlenet.png)

On [this page](https://blbadger.github.io/input-generation.html) it was observed that a gradient descent on the input was capable of producing images that were confidently predicted to be of one output category.  To test whether or not deep dream enhances characteristics of those categories or chooses new ones, these generated representative images will be used as initial inputs $a_0$.  We apply the 3-octave gradient descent without smoothness or positional jitter, meaning the update is

$$
a_{n+1} = a_n + \epsilon * g_n
$$

For some layers it is clear that certain characteristics are introduced irrespective of the original image $a_0$.  Optimizing layer 4b leads to the appearance of fur and animal faces on most representative ImageNet categories, for example on this image for 'Stoplight'

![deep dream stoplight]({{https://blbadger.github.io}}/neural_networks/googlenet_stoplight_4bdream.png)




### Directed Dream






















