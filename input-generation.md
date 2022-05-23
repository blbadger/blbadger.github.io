## Input Generation from Trained Classifiers

### Modifying an input via gradient descent

The observation that a deep learning model may be able to capture much of the inportant information in the input image leads to a hypothesis: perhaps we could use a trained model in reverse to generate inputs that resemble the training inputs.

In the previous section we saw that the loss gradient with respect to the input $\nabla_a J(O(a; \theta), y)$ of a trained model is able to capture certain features of input images: in particular, the gradient mirrors edges and certain colors that exist in the input.  This observation leads to an idea: perhaps we could use this gradient to try to make our own input image by starting with some known distribution and repeatedly applying the loss gradient to the input.  This process mirrors how stochastic gradient descent applies the loss gradient with respect to the model parameters each minibatch step, but instead of modifying the model parameters instead we are going to modify the input itself.

If we want to apply the loss gradient (of the input) to the input, we need three things: a trained model with parameters $\theta$, and input $a$, and an output $y$.  One can assign various distributions to be the input $a$, and arbitrarily we can begin with a stochastic distribution rather than a uniform one.  Next we need an output $y$ that will determine our loss function value: the close the input $a$ becomes to a target input $a'$ that the model expects from learning a dataset given some output $y$, the smaller the loss value.  We can also arbitrarily choose an expected output $\widehat{y}$ with which we use to modify the input, but for a categorical image task it may be best to choose one image label as our target output $\widehat{y}$

Each step of this process, the current input $a_n$ is modified to become $a_{n+1}$ as follows

$$
a_{n+1} = a_n - \epsilon \nabla_a J(O(a; \theta), \widehat{y})
$$

Intuitively, a trained model should know what a given output category generally 'looks like', and performing gradient-based updates on an image while keeping the output constant (as a given category) is similar to the model instructing the input as to what it should become to match the model's expectation.

To implement this algorithm, first we need a function that can calculate the gradient of the loss function with respect to the input

```python
def loss_gradient(model, input_tensor, true_output, output_dim):
	...
	true_output = true_output.reshape(1)
	input_tensor.requires_grad = True
	output = model.forward(input_tensor)
	loss = loss_fn(output, true_output) # loss applied to y_hat and y

	# backpropegate output gradient to input
	loss.backward(retain_graph=True)
	gradient = input_tensor.grad
	return gradient
```

Then the gradient update can be made at each step

```python
def generate_input(model, input_tensors, output_tensors, index, count):
	... 
	target_input = input_tensors[index].reshape(1, 3, 256, 256)
	single_input = target_input.clone()
	single_input = torch.rand(1, 3, 256, 256) # uniform distribution initialization

	for i in range(1000):
		single_input = single_input.detach() # remove the gradient for the input (if present)
		predicted_output = output_tensors[index].argmax(0)
		input_grad = loss_gradient(model, single_input, predicted_output, 5) # compute the input gradient
		single_input = single_input - 10000*input_grad # gradient descent step

```

`single_input` can then be viewed with the target label denoted

![adversarial example]({{https://blbadger.github.io}}/neural_networks/generated_daisy_nosign.png)

but the output does not really look like a daisy, or a field of daisies.  

There are a few problems with this approach.  Firstly, it is extremely slow: the gradient of the input is sometimes small, and so updating the input using a fraction of the gradient is not feasible. Instead the gradient often must be scaled up (one method to do this is to use a constat scale, perhaps `10000*input_grad`) but doing so brings no guarantee that $a_{n+1}$ will actually have a lower loss than $a_n$. 

The second problem is more pervasive: the discontinuities present in the model output $O(a; \theta)$ (see the section on adversarial examples above), which necessarily make the reverse function also discontinuous.  It has been hypothesized that adversarial examples exist in spite of the high accuracy achieved various test datasets because they are very low-probability inputs.  In some respects this is true, as the addition of a small vector in a random direction (rather than the direction of the gradient with respect to the output) very rarely changes the model's output.

![adversarial example]({{https://blbadger.github.io}}/neural_networks/flower_random_addition2.png)

But this is only the case when we look at images that are approximations of inputs that the model might see in a training or test set.  In the adversarial example approaches taken above, small shifts are made to each element (pixel) of a real image to make an approximately real image.  If we no longer restrict ourselves in this way, we will see that adversarial examples are actually much more common.  Indeed that almost any input that a standard image classification model would classify as any given label with high confidence does not resemble a real image.  For more information on this topic, see the next section.

In practice the second problem is more difficult to deal with than the first, which can be overcome with clever scaling and normalization of the gradient.  The main problem is therefore that the gradient of the loss (or the output, for that matter) with respect to the input $a$, $\nabla_a J(O(a, \theta))$ is approximately discontinuous in certain directions can cause a drop in the loss function even though the input is far from a realistic image.  The result is that more or less unrecognizable images like the one above are confidently but erroneously classified as being an example of the correct label. For more on the topic of confident but erroneous classification using deep learning, see [this paper](https://arxiv.org/abs/1412.1897).

One way to ameliorate these problems is to go back to our gradient sign method rather than to use the actual gradient.  This introduces the prior assumption that   This allows us to restrict the changes at each iteration to a constant step, stabilizing the gradient update. 

$$
a_{n+1} = a_n - \epsilon \; \mathrm {sign} \; (\nabla_a J(O(a; \theta), \widehat{y}))
$$

which for $\epsilon=0.01$ can be implemented as

```python
	...
	for i in range(100):
		single_input = single_input.detach() # remove the gradient for the input (if present)
		# predicted_output = output_tensors[index].argmax(0)
		input_grad = loss_gradient(model, single_input, target_output, 5) # compute input gradient
		single_input = single_input - 0.01*torch.sign(input_grad) # gradient descent step
```

Secondly, instead of starting with a random input we can instead start with some given flower image from the dataset.  The motivation behind this is to note that the model has been trained to discriminate between images of flowers, not recognize images of flowers compared to all possible images in $\Bbb R^n$ with $n$ being the dimension of the input.  

This method is more successful: when the target label is a tulip, observe how a base and stalk is added to a light region of an image of a field of sunflowers,

![adversarial example]({{https://blbadger.github.io}}/neural_networks/generated_tulip.png)

and how a rock is modified to appear more like a field of tulips,

![adversarial example]({{https://blbadger.github.io}}/neural_networks/generated_daisy2.png)

and likewise a daisy's features (white petals etc.) are modified to 

![adversarial example]({{https://blbadger.github.io}}/neural_networks/generated_daisy.png)

but generally images of tulips are changed less, which is to be expected given that the gradient of the loss function with respect to the input will be smaller if our target output matches our actual output.

![adversarial example]({{https://blbadger.github.io}}/neural_networks/generated_tulip_orig.png)

### Generating Images using Prior Assumptions

In the last section we saw that using the gradient of the output with respect to the input $\nabla_a J(O(a, \theta))$ may be used to modify the input in order to make it more like what the model expects a given parameter label to be.  But applying this gradient to an initial input of pure noise was not found to give a realistic representation of the desired type fo flower, because the loss function is discontinuous with respect to the input.  Instead we find a type of adversarial example which is confidently assigned to the correct label by the model, but does not actually resemble any kind of flower at all.

Is there some way we can prevent our trained deep learning models from making unrealistic images during gradient descent on a random input?  Research into this question has found that indeed there is a way: restrict the image modification process such that some quality of a real image is enforced.  We will proceed with this investigation using an Inceptionv3 (aka GoogleNetv3) trained on the full ImageNet dataset, which consists of labelled images of 1000 classes.

Which qualities of a real image should be enforced during gradient descent? A good paper on this topic by [Olah and colleages](https://distill.pub/2017/feature-visualization/) details how different research groups have attempted to restrict a variety of qualities, but most fall into fouor categories: input or gradient regularization, frequency penalization, transformational invariance, and a learned prior. 

We will focus on the first three of these restrictions in this section.  

Suppose we want to transform an original input image of noise into an image of some category for which a model was trained to discern.  We could do this by performing gradient descent using the gradient of the loss with respect to the input

$$
a' = a - \epsilon \nabla_a J(O(a, \theta))
$$

which was attempted above.  One difficulty with this approach is that many models use a softmax activation function on the final layer (called logits pre-activation) to make a posterior probability distribution, allowing for an experimentor to find the probability assigned to each class.  Minimizing $J(\mathrm{softmax} \; O(a, \theta))$ may occur via maximizing the value of $O_n(a, \theta)$ where $O_n$ is the index of the target class, the category of output we are attempting to represent in the input.  But as pointed out by [Simonyan and colleagues](https://arxiv.org/pdf/1312.6034v2.pdf), minimization of this value may also occur via minimization of $O_{-n}$, ie minimimization of all outputs at indicies not equal to the target class index.

To avoid this difficulty, we will arrange our loss function such that we maximize the value of $O_n(a, \theta)$ where $O_n$ signifies the logit at index n.  A trivial way to maximize this value would be to simply maximize the values of all logits at once, and so to prevent this we use L1 regularization (althogh L2 or other regularizers should work well too).  We can either regularize with respect to the activations of the final layer, or else with respect to the input directly.  As the model's parameters $\theta$ do not change during this gradient descent, these approaches are equivalent and so we take the latter.

The objective function we will minimize is therefore

$$
J'(a) = J(O_n(a, \theta)) + \sum_i \vert a_i \vert
$$

and for our primary loss function we also have a number of possible choices. Here we will take the L1 metric to some large constant $C$

$$
J'(a) = (C - O_n(a, \theta)) + \sum_i \vert a_i \vert
$$

and this may be implemented using pytorch as follows:

```python
def layer_gradient(model, input_tensor, true_output):
	...
	input_tensor.requires_grad = True
	output = model(input_tensor)
	loss = torch.abs(200 - output[0][int(true_output)]) + 0.001 * torch.abs(input_tensor).sum() # maximize output val and minimize L1 norm of the input
	loss.backward()
	gradient = input_tensor.grad
	return gradient
```

Note that we are expecting the model to give logits as outputs rather than the softmax values.  The pytorch version of InceptionV3 does so automatically, which means that we can simply use the output directly without having to use a hook to find the appropriate values.

In a related vein, we will also change our starting tensor allow for more stable gradients, leading to a more stable loss as well. Instead of using a uniform random distribution ${x: x \in (0, 1)}$ alone, the random uniform distribution is scaled down by a factor of 25 and centerd at $1/2$ as follows:

```python
single_input = (torch.rand(1, 3, 256, 256))/25 + 0.5 
```

Now we can use our trained `model` combined with the `layer_gradient` to retrieve the gradient of our logit loss with respect to the input, and modify the input to reduce this loss.  

```python
def generate_input(model, input_tensors, output_tensors, index, count):
	...
	class_index = 292
	single_input = (torch.rand(1, 3, 256, 256))/25 + 0.5 # scaled normal distribution initialization
 
	single_input = single_input.to(device)
	single_input = single_input.reshape(1, 3, 256, 256)
	original_input = torch.clone(single_input).reshape(3, 256, 256).permute(1, 2, 0).cpu().detach().numpy()
	target_output = torch.tensor([class_index], dtype=int)

	for i in range(100):
		single_input = single_input.detach() # remove the gradient for the input (if present)
		predicted_output = model(single_input)
		input_grad = layer_gradient(model, single_input, target_output) # compute input gradient
		single_input = single_input - 0.15 * input_grad # gradient descent step
```

It should be noted that this input generation process is fairly tricky: challenges include unstable gradients resulting learning rates (here `0.15`) being too high or initial inputs not being scaled correctly, or else the learning rate not being matchd with the number of iterations being performed.  Features like learning rate decay and gradient normalization were not found to result in substantial improvements to the resulting images.

For most ImageNet categories, the preceeding approach does not yield very recognizable images.  Features of a given category are often muddled together or dispered throughout the generated input.  Below is a typical result, in this case when 'ant' is chosen (`class_index = 310`).  The initial random image is transformed as the class activation (logit) increases for the appropriate index.

{% include youtube.html id='x5ydF_bORFQ' %}

A minority of classes do have vaguely recognizable images generated: when we use the category 'washing machine' as our target, the glass covers of side-loading washing machines are represented as small round objects in distorted rectangles.

![washer]({{https://blbadger.github.io}}/neural_networks/generated_washer.png)

The second Bayesian prior we will enforce is that images will not be too variable from one pixel to the next.  Reducing variability between adjacent pixels increases their correlation correlation with each other (see here for an account on pixel correlation [here](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)) such that that neighboring pixels are constrained to resemble each other, in effect smoothing out an image.  One way to reduce variability between nearby pixels is to perform a convolution, the same operation which our model uses judiciously in neural network form for image classification.  

To recap, a convolution (in the context of image processing) is a function in which a pixel's value is added to that of its neighbors according to some given weight.  The weights are called the kernal or filter, and are usually denoted as $\omega$ Arguably the simplest example is the uniform kernal, which in 3x3 is as follows:

$$
\omega = 
\frac{1}{9}
\begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1 \\
\end{bmatrix}
$$

To perform a convolution, this kernal is applied to each pixel in the input to produce an output image in which each pixel corresponds to the average of itself along with all adjacent pixels.  To see how this kernal blurs an input, say the following pixel values were found somewhere in an image $f(x, y)$:

$$
f(x, y) = 
\begin{bmatrix}
1 & 2 & 3 \\
0 & 10 & 2 \\
1 & 3 & 0 \\
\end{bmatrix}
$$

The convolution operation applied to this center element of the original image $f(x_1, y_1)$ with pixel intensity $10$ now is

$$
\omega * f(x_1, y_1) =
\frac{1}{9}
\begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1 \\
\end{bmatrix} 
*
\begin{bmatrix}
1 & 2 & 3 \\
0 & 10 & 2 \\
1 & 3 & 0 \\
\end{bmatrix}
$$

The two-dimensional convolution operation is computed in an analagous way to the one-dimensional vector dot (scalar) product.  Indeed, the convolutional operation is what is called an inner product, which is a generalization of the dot product into two or more dimensions.  Dot products convey information of both magnitude and the angle between vectors, and similarly inner products convey both a generalized magnitude magnitude and a kind of multi-dimensional angle between operands.  

The calculation for the convolution is as follows:

$$
\omega * f(x_2, y_2) = 1/9 (1 \cdot 1 + 1 \cdot 2 + 1 \cdot 3 + \\
1 \cdot 0 + 1 \cdot 10 + 1 \cdot 2 + 1 \cdot 1 + 1 \cdot3 + 1 \cdot 0) \\
= 22/9 \\
\approx 2.44 < 10
$$

so that the convolved output $f'(x, y)$ will have a value of $2.4$ in the same place as the unconvolved input $f(x, y)$ had value 10.  

$$ f'(x, y) = 
\begin{bmatrix}
a_{1, 1} & a_{1, 2} & a_{1, 3} \\
a_{2, 1} & a_{2, 2} = 22/9 & a_{2, 3} \\
a_{3, 1} & a_{3, 2} & a_{3, 3} \\
\end{bmatrix}
$$

Thus the pixel intensity of $f'(x, y)$ at position $m, n = 2, 2$ has been reduced.  If we calculate the other values of $a$, we find that it is more similar to the values of the surrounding pixels compared to the as well. The full convolutional operation simply repeats this process for the rest of the pixels in the image to calculate $f'(x_1, y_1), ..., f'(x_m, y_n)$.  A convolution applied using this particular kernal is sometimes called a normalized box blur, and as the name suggests it blurs the input slightly. 

But depending on the kernal, we can choose to not blur an image at all. Here is the identity kernal, which gives an output image that is identical with the input.

$$
\omega = 
\begin{bmatrix}
0 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 0 \\
\end{bmatrix}
$$

We can even sharpen an input, ie decrease the correlation between neighboring pixels, with the following kernal.

$$
\omega = 
\begin{bmatrix}
0 & -1/2 & 0 \\
-1/2 & 2 & -1/2 \\
0 & -1/2 & 0 \\
\end{bmatrix}
$$

One choice for convolution is to use a Gaussian kernal with a 3x3 size.  The Gaussian distribution has a number of interesting properties, and arguably introduces the least amount of information in its assumption.  A Gaussian distribution in two dimensions $x, y$ is as follows:

$$
G(x, y) = \frac{1}{2 \pi \sigma^2} \mathrm{exp} \left( \frac{x^2 + y^2}{2 \sigma^2} \right)
$$

where $\sigma$ is the standard deviation, which we can specify.  Using the functional Gaussian blur module of `torchvision`, the default value for a 3x3 kernal is $\sigma=0.8$ such that the kernal we will use is to a reasonable approximation

$$
\omega = 
9/4
\begin{bmatrix}
1/4 & 1/2 & 1/4 \\
1/2 & 1 & 1/2 \\
1/4 & 1/2 & 1/4 \\
\end{bmatrix}
$$

where the $1/2$ and $1/4$ values are more precisely just over $201/400$ and $101/400$.

Here we apply a Gaussian convolution to the input in a curriculum in which our convolution is applied at every iteration until three quarters the way through gradient descent, at which time it is no longer applied for the remaining iterations.  Heuristically this should a general shape to form before details are filled in.


```python
def generate_input(model, input_tensors, output_tensors, index, count):
	max_iterations = 100
	for i in range(max_iterations):
		...
		if i < (max_iterations - max_iterations/4):
			single_input = torchvision.transforms.functional.gaussian_blur(single_input, 3)
```

Choosing to optimize for a classification of 'ant' (`class_index = 310`) once again, we see that this smoothness prior leads to a more unstable class prediction but a more realistic generated image of an ant.

{% include youtube.html id='5oOgiRQfDyQ' %}

Smoothness along leads to a number of more recognizable images being formed,

![convolved array]({{https://blbadger.github.io}}/neural_networks/generated_blurred.png)

Observe that these images are somewhat dim.  This results from the presence of spikes in pixel intensity for very small regions, which the smoothing process does not completely prevent.  The rest of the images using the smoothness prior alone (but no others) displayed below have been adjusted for brightness and contrast for clarity. 

Depending on the initial input given to the model, different outputs will form.  For a target class of 'Tractor', different initial inputs give images that tend to show different aspect of a tractor: sometimes the distinctive tyre tread, sometimes the side of the wheel, sometimes the smokestack is visible.

![convolved tractor]({{https://blbadger.github.io}}/neural_networks/generated_tractor_array.png)

This is perhaps in part the result of the ImageNet training set containing images where not the entire tractor is visible at once.  This appears to be a typical result for this image generation technique: if we optimize for 'Badger', we usually see the distinctive face pattern but in a variety of orientations

![convolved badgers]({{https://blbadger.github.io}}/neural_networks/generated_badgers.png)

Here for 'Ant' we have

![convolved ant]({{https://blbadger.github.io}}/neural_networks/generated_ant.png)

although for other classes, few perspectives are reached: observe that for 'Keyboard' the geometry of the keys but not the letters on the keys are consistently generated.

![convolved keyboard]({{https://blbadger.github.io}}/neural_networks/generated_keyboard.png)

The final prior we will add is for transformational resiliency.  The idea here is that we want to generate images that the model does not classify very differently if a small transformation is applied.  This transformation could be a slight change in color, a change in image resolution, a translation or rotation, among other possibilities.  Along with a Gaussian convolution, we also apply to the first three quarters of all images one of five re-sizing transformations.

In addition, a small intensity change is applied to each pixel at random for each iteration using `torchvision.transforms.ColorJitter(c)` where `c` is a value of choice.  Specifically, $\epsilon \in [-c, c]$ is added to element $a_{x, y}$ of input $a$ to make element $a_{x, y}'= a{x, y} + \epsilon$ of a transformed input $a'$.  In the code sample below, we assign $\epsilon \in [0.0001,0.0001]$ but this choice is somewhat arbitrary.  Note that this transformation may also be undertaked with much larger values (empirically up to around $\epsilon = 0.05$) and for color, contrast, and saturation as well as brightness by modifying the arguments to `torchvision.transforms.ColorJitter()`.

```python
def generate_input(model, input_tensors, output_tensors, index, count):
	max_iterations = 100
	for i in range(max_iterations):
		...
		single_input = torchvision.transforms.ColorJitter(0.0001)(single_input)
		if i < 76:
			...
			if i % 5 == 0:
				single_input = torch.nn.functional.interpolate(single_input, 256)
			else:
				single_input = torch.nn.functional.interpolate(single_input, 100 + 28*(i % 5 - 1))
				
			# optional: resize back to 256x256
			# single_input = single_input = torch.nn.functional.interpolate(single_input, 256)
```

The class label (and input gradient) tends to be fairly unstable during training, but the resulting images can be fairly recognizable: observe the lion's chin and mane appear in the upper left hand corner during input modification.

{% include youtube.html id='yp9axdNcCG8' %}

And this is not atypical, as for half of our 16 random input we get a recognizable lion face and mane.

![generated lion]({{https://blbadger.github.io}}/neural_networks/generated_multiscalejitter_lion.png)

We can compare the effects of adding transformations to the modification process to the resulting image.  Here the convolved (such that smoothness is enforced) images are not adjusted for brightness and contrast as they were above.  The figures below demonstrates ho how re-sizing during input modification effectively prevents spikes in pixel intensity.  For a soccer ball and a tennis ball, we have

![generated soccerballs]({{https://blbadger.github.io}}/neural_networks/generated_soccerball_comparison.png)

![generated tennisballs]({{https://blbadger.github.io}}/neural_networks/generated_tennisball_comparison.png)

Some inputs are now unmistakeable, such as this badger's profile

![generated badger]({{https://blbadger.github.io}}/neural_networks/generated_singlebadger.png)

or 'Chainmail'

![generated badger]({{https://blbadger.github.io}}/neural_networks/generated_chainmail_multiscalejitter.png)

Note, however, that transformational invariance does not necessarily lead to a more recognizable image for all class types: ants, for example, are generally clearer when not transformed.

We can also add rotations and translations.  If one expects an image class to contain examples for any arbitrary angle, we can train whilst rotating the input in place.  Here we have a 'Strawberry' 

![generated strawberry]({{https://blbadger.github.io}}/neural_networks/generated_transformed_strawberry.png)
