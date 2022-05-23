## Input Attribution, Adversarial Examples, and Input Generation

### Introduction: Fashion MNIST

The [fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) is a set of 28x28 monocolor images of articles of 10 types of clothing, labelled accordingly.  Because these images are much smaller than the 256x256 pixel biological images above, the architectures used above must be modified (or else the input images must be reformatted to 256x256).  The reason for this is because max pooling (or convolutions with no padding) lead to reductions in subsequent layer size, eventually resulting in a 0-dimensional layer.  Thus the last four max pooling layers were removed from the deep network, and the last two from the AlexNet clone ([code](https://github.com/blbadger/neural-network/blob/master/fmnist_bench.py) for these networks).  

The deep network with no other modifications than noted above performs very well on the task of classifying the fashion MNIST dataset, and >91 % accuracy rate on test datasets achieved with no hyperparameter tuning. 

![fashion MNIST]({{https://blbadger.github.io}}/neural_networks/Fashion_mnist.png)

AlexNet achieves a ~72% accuracy rate on this dataset with no tuning or other modifications, although it trains much slower than the deep network as it has many more parameters (>10,000,000) than the deep network (~180,000).

We may observe a model's attribution on the inputs from this dataset as well in order to understand how a trained model arrives at its conclusion. Here we have our standard model architecture and we compute the gradientxinput

$$
\nabla_a O(a; \theta) * a
$$

where the input $a$ is a tensor of the input image (28x28), the output of the model with parameters $\theta$ and input $a$ is $O(a; \theta)$, and $*$ denotes Hadamard (element-wise) multiplication.  This may be accomplished using Pytorch in a similar manner as above, and for some variety we can also perform the calculation using Tensorflow as follows:

```python
def gradientxinput(features, label):
	...
	optimizer = tf.keras.optimizers.Adam()
	features = features.reshape(1, 28, 28, 1)
	ogfeatures = features # original tensor
	features = tf.Variable(features, dtype=tf.float32)
	with tf.GradientTape() as tape:
		predictions = model(features)

	input_gradients = tape.gradient(predictions, features).numpy()
	input_gradients = input_gradients.reshape(28, 28)
	ogfeatures = ogfeatures.reshape(28, 28)
	gradxinput = tf.abs(input_gradients) * ogfeatures
	...
```
such that `ogfeatures` and `gradxinput` may be fed directly into `matplotlib.pyplot.imshow()` for viewing.  Note that the images and videos presented here were generated using pytorch, with a similar (and somewhat less involved) implementation as the one put forth in the preceding section for obtaining gradientxinput tensors. 

For an image of a sandal, we observe the following attribution:

![fashion MNIST gradientxinput]({{https://blbadger.github.io}}/neural_networks/fmnist_gradxinput.png)

which focuses on certain points where the sandal top meets the sole.  How does a deep learning model such as our convolutional network learn which regions of the input to focus on in order to minimize the cost function?  At the start of training, there is a mostly random gradientxinput attribution for each image

![fashion MNIST gradientxinput]({{https://blbadger.github.io}}/neural_networks/fmnist_attribution_grid0001.png)

but at the end of training, certain stereotypical features of a given image category receive a larger attribution than others: for example, the elbows and collars of coats tend to exhibit a higher attribution than the rest of the garment.

![fashion MNIST gradientxinput]({{https://blbadger.github.io}}/neural_networks/fmnist_attribution_grid0505.png)

It is especially illuminating to observe how attribution changes after each minibatch gradient update.  Here we go from the start of the start to the end of the training as show in the preceding images, plotting attributions on a subset of test set images after each minibatch (size 16) update.

{% include youtube.html id='7SCd5YVYejc' %}

### Flower Type Identification

For some more colorful image classifications, lets turn to Alexander's flower [photoset](https://www.kaggle.com/alxmamaev/flowers-recognition), containing labeled images of sunflowers, tulips, dandelions, dasies, and roses.  The deep network reaches a 61 % test classification score on this dataset, which increases to 91 % for binary discrimination between some flower types. Examples of this model classifying images of roses or dandelions,

![flower classes]({{https://blbadger.github.io}}/neural_networks/Figure_flowers2.png)

sunflowers or tulips,

![flower classes]({{https://blbadger.github.io}}/neural_networks/Figure_flowers1.png)

and tulips or roses

![flower classes]({{https://blbadger.github.io}}/neural_networks/Figure_flowers_tulips_roses2.png)

We can investigate the learning process by using gradientxinput attribution. Before the start of training, we see that there is relatively random attribution placed on various pixels of test set flower images

![flower saliency]({{https://blbadger.github.io}}/neural_networks/flower_attributions0000.png)

but after 25 epochs, certain features are focused upon

![flower saliency]({{https://blbadger.github.io}}/neural_networks/flower_attributions1200.png)

Plotting attribution after every minibatch update to the gradient, we have

{% include youtube.html id='lVcNSD0viX0' %}

The deep learning models generally perform worse on flower type discrimination when they are not given color images, which makes sense being that flowers are usually quite colorful.  Before the start of training, we have a stochastic attribution: note how the model places relatively high attribution on the sky in the bottom three images (especially the bottom right)

![flower saliency]({{https://blbadger.github.io}}/neural_networks/colorflower_attributions0000.png)

In contrast, after 25 epochs of training the model has learned to place more attribution on the tulip flower body, the edge of the rose petals, and the seeds of the sunflower and dandelion.  Note that the bottom center tulip has questionable attribution: the edge of the leaves may be used to discriminate between plant types, but it is not likely that flower pot itself is a useful feature to focus on.

![flower saliency]({{https://blbadger.github.io}}/neural_networks/colorflower_attributions1202.png)

Plotting attribution after every minibatch update to the gradient, we have

{% include youtube.html id='mz_Qo1fcmgc' %}

### Adversarial Examples

Considering the attribution patterns placed on various input images, it may seem that a deep learning object recognition process is similar to a human-like decision making process when identifying images: focus on the features that differ between images and learn which features correspond to what image. But there are significant differences between natural and deep learning-based object recognition, and one of the most dramatic of these differences is the presence of what has been termed 'adversarial examples', first observed by Szegedy and colleagues in their [paper](https://arxiv.org/abs/1312.6199) on this subject.

To those of you who have read [this page](https://blbadger.github.io/nn-limitations.html) on the subject, the presence of adversarial examples should come as no surprise: as a model becomes able to discriminate between more and more input images it better and better approximates a one-to-one mapping between a multidimensional input (the image) and a one-dimensional output (the cost function).  To summarize the argument on that page, there are no continuous one-to-one (bijective) mappings possible from two or more dimensions to one, we would expect to see discontinuities in a function approximating a bijective map between many and one dimension.  This is precisely what occurs when an image tensor input (which for a 28x28 image is 784-dimensional) is mapped by a deep learning model to a loss value by $J(O(a; \theta), y)$.

How might we go about finding an adversarial example?  One option is to compute the gradient $g$ of the loss function of the output $J(O)$ with respect to the input $a$ of the model with parameters $\theta$ with a true classification $y$,

$$
g = \nabla_a J(O(a; \theta), y)
$$

but instead of taking a small step against the gradient (as would be the case if we were computing $\nabla_{\theta} J(O(a; \theta))$ and then taking a small step in the opposite direction during stochastic gradient descent), we first find the direction along each input tensor element that $g$ projects onto with $\mathrm{sign}(g)$ and then take a small step in this direction.

$$
a' = a + \epsilon * \mathrm{sign}(g)
$$

where the $\mathrm{sign}()$ function the real-valued elements of a tensor $a$ to either 1 or -1, or more precisely this function 

$$
f: \Bbb R \to \{ -1, 1 \} 
$$ 

depending on the sign of each element $a_n \in a$. This is known as the fast gradient sign method, and has been reported to yield adversarial examples for practically any CIFAR image dataset input when applied to a trained AlexNet architecture.

What this procedure accomplishes is to change the input by a small amount (determined by the size of $\epsilon$) in the direction that makes the cost function $J$ increase the most, which intuitively is effectively the same as making the input a slightly different in precisely the direction per pixel that makes the neural network less accurate.  

To implement this, first we need to calculate $g$, which may be accomplished as follows:

```python
def loss_gradient(model, input_tensor, true_output, output_dim):
	... # see source code for the full method with documentation
	true_output = true_output.reshape(1)
	input_tensor.requires_grad = True
	output = model.forward(input_tensor)
	loss = loss_fn(output, true_output) # objective function applied (negative log likelihood)

	# only scalars may be assigned a gradient
	output = output.reshape(1, output_dim).max()

	# backpropegate output gradient to input
	loss.backward(retain_graph=True)
	gradient = input_tensor.grad
	return gradient
```

Now we need to calculate $a'$, and here we assign $\epsilon=0.01$. Next we find the model's output for $a$ as well as $a'$, and throw in an output for $g$ for good measure

```python
def generate_adversaries(model, input_tensors, output_tensors, index):
	... # see source code for the full method with documentation
	single_input= input_tensors[index].reshape(1, 3, 256, 256)
	input_grad = torch.sign(loss_gradient(model, single_input, output_tensors[index], 5))
	added_input = single_input + 0.01*input_grad
	
	original_pred = model(single_input)
	grad_pred = model(0.01*input_grad)
	adversarial_pred = model(added_input)
```

Now we can plot images of $a$ and the output of each fed to the model by reshaping the tensor for use in `plt.imshow()` before finding the output class and output confidence (as we are using a softmax output) from $O(a; \theta)$ as follows

```python
	...
	input_img = single_input.reshape(3, 256, 256).permute(1, 2, 0).detach().numpy() # reshape for imshow 
	original_class = class_names[int(original_pred.argmax(1))].title() # find output class
	original_confidence = int(max(original_pred.detach().numpy()[0]) * 100) # find output confidence w/softmax output
```

and finally we can perform the same procedure to yield $g$ and $a'$.  

For an untrained model with randomized $\theta$, $O(a';\theta)$ is generally quite similar to $O(a;\theta)$. The figures below display typical results from computing $\mathrm{sign}(g)$ (center) from the original input $a$ (left) with the modified input $a'$ (right). Note that the image representation of the sign of the gradient clips negative values (black pixels in the image) and is not a true representation of what is actually added to $a$: the true image of $\mathrm{sign}(g)$ is 50 times dimmer than shown because by design $\epsilon * \mathrm{sign}(g)$ is practically invisible. The model's output for $a$ and $a'$ is noted above the image, with the softmax value converted to a percentage for clarity.

![adversarial example]({{https://blbadger.github.io}}/neural_networks/flower_start_adversarial0.png)

![adversarial example]({{https://blbadger.github.io}}/neural_networks/flower_start_adversarial1.png)

After training, however, we see some dramatic changes in the model's output (and ability to classify) the image $a$ compared to the shifted image $a'$. In the following three examples, the model's classification for $a$ is accurate, but the addition of an imperceptably small amount of the middle image to make $a'$ yields a confident but incorrect classification.

![adversarial example]({{https://blbadger.github.io}}/neural_networks/adversarial_example0.png)

![adversarial example]({{https://blbadger.github.io}}/neural_networks/adversarial_example1.png)

![adversarial example]({{https://blbadger.github.io}}/neural_networks/adversarial_example6.png)

Not all shifted images experience this change in predicted classification: the following images are viewed virtually identically by the model.  After 40 epochs of training a cNN, around a third of all inputs follow this pattern such that the model does not change its output significantly when given $a'$ as an input instead of $a$ for $\epsilon=0.01$.  Note that increasing the value of $\epsilon$ leads to nearly every input image having an adversarial example using the fast gradient sign method, even if the images are still not noticeably different.

![adversarial example]({{https://blbadger.github.io}}/neural_networks/adversarial_example13.png)

It is interesting to note that the gradient sign image itself may be confidently (and necessarily incorrectly) classified too.

![adversarial example]({{https://blbadger.github.io}}/neural_networks/gradpred_adversarial_example3.png)

Can we find adversarial examples for simpler inputs as well as complicated ones? Indeed we can: after applying the gradient step method to 28x28 pixel Fashion MNIST images using a model trained to classify these inputs, we can find adversarial examples just as we saw for flowers.

![adversarial example]({{https://blbadger.github.io}}/neural_networks/fmnist_adversarial_example.png)

It may see strange to take the sign of the gradient per pixel rather than the projection of the gradient itself, as would be the case if $a$ were a trainable parameter during gradient descent. The authors [this work](https://arxiv.org/pdf/1412.6572.pdf) made this decision in order to emphasize the ability of a linear transformation in the input to create adversarial examples, and went on to assert that the major cause of adversarial examples in general is excessive linearity in deep learning models.

It is probable that such linearity does indeed make finding adversarial examples somewhat easier, but if the argument on [this page](https://blbadger.github.io/nn-limitations.html) is accepted then attempting to prevent adversarial examples using nonlinear activation functions or specialized architectures is bound to fail, as all $f: \Bbb R^n \to \Bbb R$ are discontinuous if bijective.  Not only that, but such $f$ are everywhere discontinuous, which is why each input image will have an adversarial example if we assume that $f$ approximates a bijective function well enough.

What happens when we manipulate the image according to the gradient of the objective function, rather than its sign?  Geometrically this signifies taking the projection of the gradient $g$

$$
g = \nabla_a J(O(a; \theta), y)
$$

onto each input element $a_n$, which tells us not only which pixels to modify but also how much to modify them. When we scale this gradient by max norming

$$
g' = \frac{g}{\mathrm{max} (g)}
$$

and then applying this normed gradient to the input $a$ to make $a'$

$$
a' = a + \epsilon * g'
$$

we once again find adversarial examples, even with very small $\epsilon$.  Here the gradient tensor image is to scale: the center image is $\epsilon *g'$ and has values that are not scaled up as they were in the other images on this page (and the prediction for $a$ on the left is followed by the true classification for clarity)

![adversarial example]({{https://blbadger.github.io}}/neural_networks/continuous_adversarial_example.png)

Empirically this method performs as well if not better than the fast gradient sign procedure with respect to adversarial example generation: while keeping $\epsilon$ small enough to be unnoticeable, the majority of inputs may be found to have corresponding adversarial examples.

It is interesting to observe the gradient images in more detail: here we have the continuous gradient $\epsilon * g'$ scaled to be 60 times brighter (ie larger values) than $\epsilon * g'$ for clarity.

![adversarial example]({{https://blbadger.github.io}}/neural_networks/enhanced_continuous_adversarial_example.png)

Close inspection of the image of $\epsilon * g'$ reveals something interesting: the gradient tensor appears to mirror a number of the features of the original input image, except with dark blue petals instead of white and a mix of blue and yellow where the disk flourets (center yellow parts) are found in the original image. This may be thought of as a recapitulation of the features of an input image itself from the sole informational content of the gradient of the loss function with respect to the input,

$$
\nabla_a J(O(a; \theta), y)
$$

Recalling how forward propegation followed by backpropegation is used in order to compute this gradient, we find that these features remain after nearly two dozen vector arithmetic operations, none of which are necessarily feature-preserving.  From an informational perspective, one can think of this as the information from the input being fed into the neural network, stored as activations in the network's various layers, before that information is then used to find the gradient of the loss function with respect to the input.

The above image is not the only input that has features that are recapitulated in the input gradient: here some tulips 

![adversarial example]({{https://blbadger.github.io}}/neural_networks/adversarial_gen_tulip.png)

and a butterfly on a dandelion 

![adversarial example]({{https://blbadger.github.io}}/neural_networks/adversarial_gen_butterfly.png)

and the same is found for a daisy.

![adversarial example]({{https://blbadger.github.io}}/neural_networks/adversarial_gen_daisy.png)

It is important to note that untrained models are incapable of preserving practically any input features in the input gradient.  This is to be expected given that the component operations of forward and backpropegation have no guarantee to preserve any information.  

### Additive Attributions for Non-Locality

In the last section, we saw that the training process (here 40 epochs) leads to a preservation of certain features of the input image in the gradient of the input with respect to the loss function.  We can observe the process of feature preservation during model training as follows:

{% include youtube.html id='sflMrJLlb0g' %}

Gradientxinput has been criticized for relying entirely on locality: the gradient of a point in multidimensional space is only accurate in an infinitesmal region around that point by definition.  Practically this means that if an input were to change substantially, a pure gradient-based input attribution method may not be able to correctly attribute that change to the output (or loss function) if there is not a local-to-global equivalence in the model in question.

There are a number of ways to ameliorate this problem.  One is to directly interfere with (occlude) the input, usually in some fairly large way before observing the effect on the output.  For image data, this could mean zeroing out all pixels in a given region that scans over the entire input.  For sequential data as seen [here](https://blbadger.github.io/nn_interpretations.html), successive characters can be modified as the model output is observed.  Occlusion usually introduces substantial changes from the original input meaning that the observed output changes are not the result of local changes.  Occlusion can be combined with gradientxinput to make a fairly robust attribution method.

Another way to address locality is to add up gradients as the input is formed in a straight-line path from some null reference, an approach put forward in [this paper](https://arxiv.org/abs/1703.01365) by Yan and colleages.  More concretely, a blank image may serve as a null reference and the true image may be formed by increasing brightness (our straight-line path) until the true image is recovered.  At certain points along this process, the gradients of the model with respect to the input may be added to make one integrated gradient measurment. This method has some benefits but also has a significant downside: for many types of input, there is no clear straight-line path.  Image input data has a couple clear paths (brightness and contrast) but discrete inputs such as language encodings do not.

An alternative to this approach could be to integrate input gradients but instead of varying inputs for a trained network, we integrate the input gradients during training for one given input $a$.  If we were to use the loss gradient, for each configuration $\theta_n$ of the model during training we have

$$
g = \sum_{n} \nabla_a J(O(a; \theta_n), y)
$$

This method may be used for any input type, regardless of an ability to transform from a baseline.

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
\omega * f(x_1, y_1) = 1/9 (1 \cdot 1 + 1 \cdot 2 + 1 \cdot 3 + \\
1 \cdot 0 + 1 \cdot 10 + 1 \cdot 2 + 1 \cdot 1 + 1 \cdot3 + 1 \cdot 0) \\
= 22/9 \approx 2.44 < 10
$$

so that the convolved output $a$ will have a value of $2.4$ in the same place as the unconvolved input $f(x, y)$ had value 10

$$ a = 
\begin{bmatrix}
a_{1, 1} & a_{1, 2} & a_{1, 3} \\
a_{2, 1} & a_{2, 2} = 2.4 & a_{2, 3} \\
a_{3, 1} & a_{3, 2} & a_{3, 3} \\
\end{bmatrix}
$$

This means that the in pixel intensity of $f(x, y)$ has been reduced.  If we calculate the other values of $a$, we find that it is more similar to the values of the surrounding pixels compared to the as well. The full convolutional operation simply repeats this process for the rest of the pixels in the image to calculate $a_{1, 1}, ..., a{n, m}$.  A convolution applied using this particular kernal is sometimes called a normalized box blur, and as the name suggests it blurs the input slightly. 

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
