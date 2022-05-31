## Image Generation with Classification Models

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

### Generating Images using a Smoothness Prior

In the last section we saw that using the gradient of the output with respect to the input $\nabla_a J(O(a, \theta))$ may be used to modify the input in order to make it more like what the model expects a given parameter label to be.  But applying this gradient to an initial input of pure noise was not found to give a realistic representation of the desired type fo flower, because the loss function is discontinuous with respect to the input.  Instead we find a type of adversarial example which is confidently assigned to the correct label by the model, but does not actually resemble any kind of flower at all.

Is there some way we can prevent our trained deep learning models from making unrealistic images during gradient descent on a random input?  Research into this question has found that indeed there is a way: restrict the image modification process such that some quality of a real image is enforced.  We will proceed with this investigation using an Inceptionv3 (aka GoogleNetv3) trained on the full ImageNet dataset, which consists of labelled images of 1000 classes.

The inception architecture was designed to allow for deeper networks without an exorbitant increase in the number of parameters used.  InceptionV3 is a slight reformulation of the original architecture, and as published [here](https://arxiv.org/pdf/1512.00567v3.pdf) and has the following architecture:

![adversarial example]({{https://blbadger.github.io}}/neural_networks/inceptionv3_architecture.png)

(image source: Google [docs](https://cloud.google.com/tpu/docs/inception-v3-advanced))

After loading the model with `model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True).to(device)`, we can use the following function

```python

def count_parameters(model):
	table = PrettyTable(['Modules', 'Parameters'])
	total_params = 0
	for name, parameter in model.named_parameters():
		if not parameter.requires_grad:
			continue
		param = parameter.numel()
		table.add_row([name, param])
		total_params += param 
	print (table)
	print (f'Total trainable parameters: {total_params}')
	return total_params
```
to find that this is a fairly large model with `27161264`, or just over 27 million, trainable parameters.

```python
+--------------------------------------+------------+
|               Modules                | Parameters |
+--------------------------------------+------------+
|      Conv2d_1a_3x3.conv.weight       |    864     |
|       Conv2d_1a_3x3.bn.weight        |     32     |
|        Conv2d_1a_3x3.bn.bias         |     32     |
|      Conv2d_2a_3x3.conv.weight       |    9216    |
|       Conv2d_2a_3x3.bn.weight        |     32     |
|        Conv2d_2a_3x3.bn.bias         |     32     |
... (shortened for brevity)
|              fc.weight               |  2048000   |
|               fc.bias                |    1000    |
+--------------------------------------+------------+
```

Now that we have our trained model, which qualities of a real image should be enforced during gradient descent? A good paper on this topic by [Olah and colleages](https://distill.pub/2017/feature-visualization/) details how different research groups have attempted to restrict a variety of qualities, but most fall into fouor categories: input or gradient regularization, frequency penalization, transformational invariance, and a learned prior. 

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

For most ImageNet categories, the preceeding approach does not yield very recognizable images.  Features of a given category are often muddled together or dispered throughout the generated input.  Below is a typical result, in this case when 'ant' is chosen (`class_index = 310`).  The initial random image is transformed as the class activation (logit) increases for the appropriate index.  Logits for the model (InceptionV3) given the input to the left are displayed as a scatterplot, with the desired output in red.

{% include youtube.html id='x5ydF_bORFQ' %}

A minority of classes do have vaguely recognizable images generated: when we use the category 'washing machine' as our target, the glass covers of side-loading washing machines are represented as small round objects in distorted rectangles.

![washer]({{https://blbadger.github.io}}/neural_networks/generated_washer.png)

The second Bayesian prior we will enforce is that images will not be too variable from one pixel to the next.  Reducing variability between adjacent pixels increases their correlation correlation with each other (see here for an account on pixel correlation [here](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)) such that that neighboring pixels are constrained to resemble each other, in effect smoothing out an image.  One way to reduce variability between nearby pixels is to perform a convolution, the same operation which our model uses judiciously in neural network form for image classification.  For an introduction to convolutions in the context of deep learning, see [this page](https://blbadger.github.io/neural-networks.html#convolutions-explained)

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

![convolved badgers]({{https://blbadger.github.io}}/neural_networks/generated_badger_array.png)

Here for 'Ant' we have

![convolved ant]({{https://blbadger.github.io}}/neural_networks/generated_ant.png)

although for other classes, few perspectives are reached: observe that for 'Keyboard' the geometry of the keys but not the letters on the keys are consistently generated.

![convolved keyboard]({{https://blbadger.github.io}}/neural_networks/generated_keyboard.png)

### Generating Images using Transformation Resiliency

Next we will add transformational resiliency.  The idea here is that we want to generate images that the model does not classify very differently if a small transformation is applied.  This transformation could be a slight change in color, a change in image resolution, a translation or rotation, among other possibilities.  Along with a Gaussian convolution, we also apply to the first three quarters of all images one of five re-sizing transformations.

Re-sizing may be accomplished using the `torch.nn.functional.interpolate()` module, which defaults to a interpolation mode of nearest neighbors.  This is identical to the k-nearest neighbors algorithm with a value of $k=1$ fixed.  For images, the value of the center of a pixel is taken to be the value across the area of the entire pixel such that the new pixel's center always lies inside one pixel or another (or on a border).  To make this clearaer, suppose we were down-sampling an image by a factor of around 1.5. For the new pixel centered on a red dot for clarity, there is

![interpolation explanation]({{https://blbadger.github.io}}/neural_networks/interpolation_explanation2.png)

In addition, a small intensity change is applied to each pixel at random for each iteration using `torchvision.transforms.ColorJitter(c)` where `c` is a value of choice.  Specifically, $\epsilon \in [-c, c]$ is added to element $a_{x, y}$ of input $a$ to make element $a_{x, y}'= a{x, y} + \epsilon$ of a transformed input $a'$.  In the code sample below, we assign $\epsilon \in [0.0001,0.0001]$ but this choice is somewhat arbitrary.  Note that this transformation may also be undertaked with much larger values (empirically up to around $\epsilon = 0.05$) and for color, contrast, and saturation as well as brightness by modifying the arguments to `torchvision.transforms.ColorJitter()`.

One note of warning: `torchvision.transforms.ColorJitter()` is a difficult method for which to set a deterministic seed. `random.seen()`, `torch.set_seed()`, and `np.seed()` together are not sufficient to make the color jitter deterministic, so it is not clear how exact reproduction would occur using this method. If a deterministic reproduction is desired, it may be best to avoid using this module.

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
			elif i % 5 == 1:
				single_input = torch.nn.functional.interpolate(single_input, 198)
			elif i % 5 == 2:
				single_input = torch.nn.functional.interpolate(single_input, 160)
			elif i % 5 == 3:
				single_input = torch.nn.functional.interpolate(single_input, 180)
			elif i % 5 == 4:
				single_input = torch.nn.functional.interpolate(single_input, 200)
```

The class label (and input gradient) tends to be fairly unstable during training, but the resulting images can be fairly recognizable: observe the lion's chin and mane appear in the upper left hand corner during input modification.

{% include youtube.html id='yp9axdNcCG8' %}

There is a noticeable translation in the image from the top left to the bottom right during the initial 75 iterations with the function above.  This is the result of using nearest neighbor interpolation while resampling using `torch.nn.functional.interpolate`.  It appears that in the case where multiple pixels are equidistant from the target, the upper left-most pixel is chosen as the nearest neighbor.  When downsampling an input, we are guaranteed to have four pixels that are nearest neighbors to the new pixels.

This can be remedied by choosing an interpolation method that averages over pixels rather than picking only one pixel value.  The following diagram illustrates the upper-left nearest neighbor interpolation method compared to a bilinear interpolation. Note how the upper left pixel becomes centered down and to the right of its original center for nearest neighbor interpolation, whereas in contrast the bilinear interpolation simply averages all pixels together and does not 'move' the top left value to the center.

![interpolation explanation]({{https://blbadger.github.io}}/neural_networks/interpolation_explanation.png)

We therefore have a method that effectively translates as well as resizes our original images, which is applied in addition to the Gaussian convolution and jitter mentioned above. For around half of our 16 random input we get a recognizable lion face and mane, and for others a paw is found

![generated lion]({{https://blbadger.github.io}}/neural_networks/generated_multiscalejitter_lion.png)

We can compare the effects of adding transformations to the modification process to the resulting image.  Here the convolved (such that smoothness is enforced) images are not adjusted for brightness and contrast as they were above.  The figures below demonstrates ho how re-sizing during input modification effectively prevents spikes in pixel intensity.  For a soccer ball and a tennis ball, we have

![generated soccerballs]({{https://blbadger.github.io}}/neural_networks/generated_soccerball_comparison.png)

![generated tennisballs]({{https://blbadger.github.io}}/neural_networks/generated_tennisball_comparison.png)

Some inputs are now unmistakeable, such as this badger's profile

![generated badger]({{https://blbadger.github.io}}/neural_networks/generated_singlebadger.png)

or 'Chainmail'

![generated badger]({{https://blbadger.github.io}}/neural_networks/generated_chainmail_multiscalejitter.png)

Note, however, that transformational invariance does not necessarily lead to a noticeable increase in recognizability for all class types: ants, for example, are not as substantially improved when we use transformational invariance 

![generated ant]({{https://blbadger.github.io}}/neural_networks/generated_ant_transformed.png)

We can also add rotations and translations to our jitter and convolutions and interpolations.  If one expects an image class to contain examples for any arbitrary angle, we can train whilst rotating the input in place.  Here we have a 'Strawberry' 

![generated strawberry]({{https://blbadger.github.io}}/neural_networks/generated_transformed_strawberry.png)

### Image Transfiguration

It is worth appreciating exactly what we were able to do in the last section.  Using a deep learning model trained for image classification combined with a few general principles of how natural images should look, we were able to reconstruct a variety of recognizable images representing various desired classes.  

This is remarkable because the model in question (InceptionV3) was not designed or trained to generate anything at all, merely to make accurate classifications.  Moreover, the initial input being used as a baseline for image generation (a scaled uniform or normal random distribution) is quite different from anything that exists in the training dataset Imagenet, as these are all real images.  What happens if we start with a real image and then apply our input gradient descent method to that?  This process will be termed 'transfiguration' on this page to avoid confusion with 'transformation', which is reserved for the jitters, interpolations, rotations, and translations applied to an input.

To begin with, it may be illuminating to perform a control experiment in which the input is the same as the targeted class.  In this case we would expect to simply see an exaggeration of the features that distinguish the object of the class compared to other classes.  Applying our transformation-resistatant and Gaussian-convolved input gradient method to images of dalmations that are not found in the original Imagenet training dataset, we have

![transfigured dalmatian]({{https://blbadger.github.io}}/neural_networks/transformed_dalmatian_dalmatian.png)

The dalmatian's spots are slightly exaggerated, but aside from some general lack of resolution the dalmatians are still clearly visible. Now let's make a relatively small transfiguration from one breed of dog to another.  Beginning again with images of dalmatians but this time performing the input gradient procedure with a target class of 'Siberian Husky' we have

![transfigured dalmatian]({{https://blbadger.github.io}}/neural_networks/transformed_dalmatian_husky.png)

The spots have all but disappeared, replaced by thicker fur and the grey stripes typical of Huskies.  Note how even smaller detailes are changed: in the bottom right, note how the iris color changes from dark brown to light blue, another common Husky characteristic.

We can view the difference between a Husky and a Dalmatian according to the model by observing what changes as our target class shifts from 'Husky' to 'Dalmatian', all using a picture of a dalmatian as an input.  To do this we need to be able to gradually shift the target from the 'Husky' class (which is $\widehat y_{250}$ in ImageNet) to the 'Dalmatian' class, corresponding to $\widehat y_{251}$.  This can be accomplished by assigning the loss $J(0(a, \theta))$ $q$ maximum interations, at iteration number $n$ as follows:

$$
J_n(O(a, \theta)) = \left( c - \widehat y_{250} * \frac{q-n}{q} \right) + \left( c - \widehat y_{251} * \frac{n}{q} \right) + L_1
$$

where $L-1$ is the manhattan metric regularizer, applied to either the input directly or the output.  Applied to the input, the regularizer is as follows:

$$
L_1 (a) = \sum_i \lvert a_i \rvert
$$

Using this method, we go from $(\widehat y_250, \widehat y_{251} = (c, 0)$ to $(\widehat y_{250}, \widehat y_251 = (0, c)$ as $n \to q$.  The intuition behind this approach is that $(\widehat y_{250}, \widehat y_{251} = (c/2, c/2)$ or any other linear combination of $c$ should provide a mix of characteristics between target classes.  After running some experiments, we see that this is indeed the case: observe how the fluffy husky tail becomes thin, dark spots form on the fur, and the eye color darkens as $n$ increases.

{% include youtube.html id='1bdpG1caKMk' %}

{% include youtube.html id='PBssSJoLOhU' %}


Transforming an input from one breed of dog to another may not seem difficult, but the input gradient procedure is capable of some very impressive changes.  Here we begin with images of flowers and target the 'Castle' class

![transfigured flowers]({{https://blbadger.github.io}}/neural_networks/transformed_flowers_castle.png)

We can get an idea of how this process works by observing the changes made to the original image as gradient descent occurs.  Here over the 100 iterations of transfiguration from a flower ('flowerpot' class in green on the scatterplot in the right) to a 'Castle' target class (the red dot on the right)

{% include youtube.html id='Q1BOOJnY9iM' %}

Even with as substantial a change as a flower to a castle some outputs are unmistakable, such as this castle tower.

![transfigured flowers]({{https://blbadger.github.io}}/neural_networks/single_castle.png)

Other transfigurations are possible, such as this badger from a rose bush

![transformed flowers]({{https://blbadger.github.io}}/neural_networks/flower_badger_single.png)

or this tulip bed into a 'Tractor'

![transformed flowers]({{https://blbadger.github.io}}/neural_networks/rose_into_tractor2.png)

or these flowers transfigured into 'Soccer ball".

![transformed flowers]({{https://blbadger.github.io}}/neural_networks/transformed_flowers_soccerball.png)

Earlier it was noted that image resizing with `torch.nn.functional.interpolate()` leads to translation when downsampling the input. This can be avoided by switching the interpolation mode away from nearest neighbor, to instead average either using a bilinear or bicubic method.  This can be done in `torch.nn.functional.interpolate()` by specifying the correct keyword argument, or else we can make use of the module `torchvision.transforms.Resize()`, which defaults to bilinear mode.  The latter is the same method we used to import our images, and was employed below.  Notice how there is now no more translation in the input image.

![transfigured flowers]({{https://blbadger.github.io}}/neural_networks/transformed_flowers_strawberry.png)

### Input Generation with Auxiliary Outputs

We have seen how representatives of each image class of the training dataset may be generated using gradient descent on the input with the addition of a few reasonable priors, and how this procedure is also capable of transforming images of one class to another.  Generation of an image matching a specific class requires an output layer trained to perform this task, and for most models this means that we are limited to one possible layer.  But InceptionV3 is a somewhat unique architecture in that it has another output layer, called the auxiliary output, which is employed during training to stabilize gradients and then deactivated during evaluation with $model.eval()$.

Let's investigate whether we can perform gradient descent to generate images using this auxiliary output rather than the usual output layer.  The architecture we want to use is

![Inception Architecture]({{https://blbadger.github.io}}/neural_networks/inception_aux.png)

Note that the Pytorch implementation of the above architecture does not include the softmax layer on either the auxiliary or main output.

Pytorch uses an approach to automatic differentiation called symbol-to-number differentiation, which is also employed by Caffe.  In this approach, the derivatives (in the form of Jacobian matricies) of each layer are specified at the start of runtime, and computed to form numerical values for each node upon forward and backpropegation.  This approach is in contrast to symbolic differentiation, in which extra nodes are added to each model node (ie layer) that provides a symbolic description of the derivatives of those nodes.  The computational graph for performing backpropegation is identical between these two approaches, but the former hides the graph itself and the latter exposes it at the expense of extra memory.  The latter approach is taken by libraries like Theano and Tensorflow.

We wish to back-propegate from a leaf node of our model, but this node is not returned as an output during evaluation mode. Switching to training mode is not an option because the batch normalization layers of the model will attain new parameters, interfering with the model's ability to classify images.  

In symbolic differentiation -based libraries, computing the gradient of the input with respect to a layer that is not the output is relatively straighforward: the output and loss from the specific layer of interest are specified, and backpropegation can then proceed directly from that layer. But in Pytorch this is not possible, as all we would have from an internal node is a number representing the gradient with respect to the output, rather than instruction for obtaining a gradient from that node itself.

So instead we must modify our model itself.  If we were not using an open-source model this would not be feasible, but as we are indeed using a freely accessible model we can do this a number of different ways.  One way is to make a new class that inherits from `nn.Module` and is passed as an initialization argument the original Inceptionv3 model.  We then provide a new `forward()` method that modifies the original inceptionv3 method (located [here](https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py)) such that the layer of choice is returned.

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
		# N x 80 x 73 x 73
		x = self.model.Conv2d_4a_3x3(x)
		# N x 192 x 71 x 71
		x = self.model.maxpool2(x)
		# N x 192 x 35 x 35
		x = self.model.Mixed_5b(x)
		# N x 256 x 35 x 35
		x = self.model.Mixed_5c(x)
		# N x 288 x 35 x 35
		x = self.model.Mixed_5d(x)
		# N x 288 x 35 x 35
		x = self.model.Mixed_6a(x)
		# N x 768 x 17 x 17
		x = self.model.Mixed_6b(x)
		# N x 768 x 17 x 17
		x = self.model.Mixed_6c(x)
		# N x 768 x 17 x 17
		x = self.model.Mixed_6d(x)
		# N x 768 x 17 x 17
		x = self.model.Mixed_6e(x)
		# N x 768 x 17 x 17
		aux = self.model.AuxLogits(x)
		return aux
```

One important thing to note is that the Inceptionv3 module is quite flexible with regards to the input image size when used normally in evaluation mode, but now that we have modified the network it must be given images of size 299x299 or the dimensions of the hidden layers will not align with what is required.  This can be enforced by something as straightforward as

```python
image = torchvision.transforms.Resize(size=[299, 299])(image)
```
in the `__getitem__()` special method of our data loader (see the source code for more information). 

The gradients in this layer also tend to be slightly smaller than the original output gradients, so some re-scaling of our gradient descent constant $\epsilon$ is necessary. For clarity, at each iteration of our descent starting with input $a_n$, we make input $a_{n+1}$ by finding the L1-normalized loss of the auxiliary output layer $O'(a; \theta)$ with respect to the desired output $\widehat{y}$,

$$
a_{n+1} = a_n - \epsilon \nabla_a J(O'(a; \theta), \widehat{y})
$$

The results are interesting: perhaps slightly clearer (ie higher resolution) than when we used the normal output layer, but maybe less organized as well.

![Inception Architecture]({{https://blbadger.github.io}}/neural_networks/auxiliary_flowers_array.png)

### Octaves 

The generated images shown so far on this page exhibit to some extent or another the presence of high-frequency patterns, which can be deleterious to the ability to make an image that is accurate to a real-world example.  High frequency between image pixels often appears as areas of bright dots set near each other, or else sometimes as dark lines that seem to overlay light regions.  The wavy, almost ripple-like appearance of some of the images above appears to be the result of application of smoothing (via Gaussian kernal convolution or re-sizing) to the often chaotic and high-frequency gradient applied to the images during generation.

One way to address the problem of high frequency and apparent chaos in the input gradient during image generation is to apply the gradient at different scales. This idea was pioneered in the context of feature visualization by Mordvintsev and colleages and published in the [Deep dream](https://www.tensorflow.org/tutorials/generative/deepdream) tutorial, and is conceptually fairly straightforward: one can observe that generated images (of both features or target classes) have small-scale patterns and large-scale patterns, and often these scales do not properly interact.  When a lack of interaction occurs, the result is smaller versions of something that is found elsewhere in the images, but in a place that reduces image clarity.

### GoogleNet

It is worth noting again that the process of enforcing the priors for natural images as we are doing on this page brings into question what the images generated mean.  















