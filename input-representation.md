## Input Generation III: Model Similarity and Autoencoding Representations

### Model Similarity Vector Space

In [part I](https://blbadger.github.io/input-generation.html#inceptionv3-and-googlenet-compared), it was observed that different models generate slightly different representations of each ImageNet input class.  It may be wondered whether or not we can use the gradients of two models to make an input, perhaps by combining the gradients of a model with parameters $\theta_1$ and another with parameters $\theta_2$ to make a gradient $g'$

$$
g' = \nabla_a \left( d(C - O_i(a, \theta_1)) + e(C - O_i(a, \theta_2)) \right)
$$

where $d, e$ are constants used to scale the gradients of models $\theta_1, \theta_2$ appropriately.  For GoogleNet and Resnet, we approximate $d = 2, e = 1$ by applying the constant only to $C$ as follows (note that there is little difference in applying the constant to the gradient or the constant alone)

```python
def layer_gradient(model, input_tensor, desired_output):
    ...
    loss = 0.5*((200 - output[0][int(desired_output)]) + (400 - output2[0][int(desired_output)]))
    ...
```

![googlenet and resnet input]({{https://blbadger.github.io}}/neural_networks/combined_mousetrap.png)

Images of all 1000 ImageNet classes generated using the combined gradient of GoogleNet and ResNet are available [here](https://drive.google.com/drive/folders/1mhj8vBm02Fd6QkQwEQ4jhI9yq34ZOtR0?usp=sharing). From these images it is clear that the combined gradient is as good as or superior to the gradients from only ResNet or GoogleNet with respect to producing a coherent input image, which suggests that the gradients from these models are not substantially dissimilar.

The above observation motivates the following question: can we attempt to understand the differences between models by generating an image representing the difference between the output activations for any image $a$? 

It is also apparent that the similarities and differences in model output may be compared by viewing the output as a vector space.  Say two models were to give very similar outputs for a representation of one ImageNet class but different outputs for another class. The identities of the classes may help inform an understanding of the the difference between models.

### Model Merging

In the last section it was observed that we can understand some of the similarities and differences between models by viewing the output as a vector space, with each model's output on each ImageNet representation being a point in this space.

What if we want one model to generate a representation of an ImageNet class that is similar to another model's representation?  We have already seen that some models (GoogleNet and Resnet) generally yield recognizable input representations whereas others (InceptionV3) yield somewhat less-recognizable inputs.  But if we were stuck with only using InceptionV3 as our image-generation source, can we try to use some information present from the other models in order to generate a more-recognizalbe image?

One may hypothesize that it could be possible to train one model to become 'more like' another model using the output of the second as the target for the output of the first.  Consider one standard method of training using maximum likelihood estimation via minimization of cross-entropy betwen the true output $Q$ and the model output $Q$ given input $x$,

$$
J(O(a, \theta), \widehat y) = H(P, Q) = -\Bbb E_{x \sim P} \log Q(x)
$$

The true output $Q$ is usually denoted as a one-hot vector, ie for an image that belongs to the first ImageNet category we have $[1, 0, 0, 0...]$.  A cross-entropy loss function measures the similarity between the output distribution model output $Q$ and this one-hot tensor $Q$.  

But there are indications that this is not an optimal training loss.  Earlier on this page we have seen that for trained models, some ImageNet categories are more similar with respect to the output activations $O(a, \theta)$ to some other ImageNet categories, and different from others.  These similarities moreover are by a nearest neighbors graphical representation intuitive for a human observer, meaning that it is indeed likely that some inputs are more similar than others.

Training requires the separation of the distributions $P_1, P_2, ... P_n$ for each imagenet category $n$ in order to make accurate predictions. But if we have a trained model that has achieved sufficient separation, the information that the model has difficulty separating certain images from others (meaning that these are more similar ImageNet categories by our output metric) is likely to be useful information in training a model.  This information is not likely to be found prior to training, and thus is useful for transfer learning.


These observations motivate the hypothesis that we may be able to use the information present in the output of one model $\theta_1$ to train another model $\theta_2$ to be able to represent an input in a similar way to $\theta_1$.  More precisely, one can use gradient descent on the parameters of model 1, $\theta_1$ to make some metric between the outputs $m(O(a, \theta_1), O(a, \theta_2))$ as small as possible. In the following example, we seek to minimize  the sum-of-squares residual as our metric

$$
J(O(a, \theta_1) = \sum_i \left(O_i(a, \theta_1) - O_i(a, \theta_2) \right)^2 
$$

which can be implemented as

```python
def train(model, input_tensor, target_output):
    ...
    output = model(input_tensor)
    loss = torch.sum(torch.abs(output - target_output)**2) # sum-of-squares loss
    optimizer.zero_grad() # prevents gradients from adding between minibatches
    loss.backward()
    optimizer.step()
    return 
```
Note that more commonly-used metrics like $L^1$ or $L^2$ empirically do not lead to significantly different results as our RSS metric here.  Whichever objective function is chosen, it can be minimized by gradient descent on the model parameters given an input image $a$ is

$$
\theta_1'= \theta_1 + \epsilon * \nabla_{\theta_1} J(O(a, \theta_1))
$$

which may be implemented as

```python
def gradient_descent():
    optimizer = torch.optim.SGD(resnet.parameters(), lr=0.00001)
    # access image generated by GoogleNet
    for i, image in enumerate(images):
        break 
    image = image[0].reshape(1, 3, 299, 299).to(device)
    target_output = googlenet(image).detach().to(device)
    target_tensor = resnet(image) - target_output
    for _ in range(1000):
        train(resnet, image, target_output)
```

Gradient descent is fairly effective at reducing a chosen metric between ResNet $\theta_1$ and GoogleNet $\theta_2$.  For example, given $a$ to be GoogleNet's representation of Class 0 (Tench), $J(O(a, \theta_1))$ decreases by a factor of around 15.  But the results are less than impressive: applying gradient descent to ResNet's parameters as $\theta_1$ to reduce the sum-of-squares distance between this model's output and GoogleNet's output does not yield a model that is capable of accurately representing this class.

It may be hypothesized that this could be because although ResNet's outputs match GoogleNet's for this class, each class has a different 'meaning', ie latent space location, which would undoubtedly hinder our efforts here.  But even if we repeat this procedure to train ResNet's outputs match GoogleNet's(for that model's representations) for all 1000 ImageNet classes, we still do not get an accurate representation of Class 0 (or any other class of interest).

![resnet trained to be googlenet tench]({{https://blbadger.github.io}}/neural_networks/resnet_trained_to_be_googlenet.png)

It is certainly possible that this method would be much more successful if applied to natural images rather than generated representations.  But this would go somewhat against the spirit of this hypothesis because natural images would bring with them new information that may not exist in the models $\theta_1, \theta_2$ even if the images come from the ImageNet training set.

These results suggest that modifying $\theta_1$ to yield and output that matches that of $\theta_2$ is a somewhat difficult task, at least if we limit ourselves to no information that is not already present in the model.  But this could simply be due to known difficulties in model training rather than an inability to explore the output latent space.  

Therefore a more direct approach to making one model yield another model's representations: rather than modifying the first model's parameters $\theta_1$ via gradient descent, we instead find the coordinates of the second model's representation of a given class in our output space and then use these coordinates as a target for gradient descent on an initially randomized input $a_0$.

For clarity, this procedure starts with an image representing the representation of a certain class with respect to some model $\theta_2$, denoted $a_{\theta_2}$.  We want to have a different model $\theta_1$ learn how to generate an approximation of this representation with no outside information. First we find a target vector $\widehat{y} \in \Bbb R^n$, with $n$ being the number of possible outputs, defined as

$$
\widehat{y} = O(a_{\theta_2}, \theta_1) 
$$

or in words the target vector is the output of the model of interest $\theta_1$ when the input supplied is the representation of some class for the model we want to emulate.

Now rather than performing gradient descent using a target $C \in \Bbb R^n$ being a vector with a large constant in the appropriate index for our class of interest and zeros elsewhere, we instead try to reduce the distance between $O(a_0, \theta_1)$ and the target vector $\widehat{y}$. Again this distance can be any familiar metric, perhaps $L^1$ which makes the computation as follows:

$$
g = \nabla_a \sum_n \lvert \widehat{y_n} - O_n(a, \theta_1) \rvert \\
$$

with the input modification being a modified version of gradient descent using smoothness (ie pixel cross-correlation) via Gaussian convolution $\mathcal{N}$ and translational invariance with Octave-based jitter, here denoted $\mathscr{J}$, were employed with the gradient for input $a_n$, denoted $g_n$.  The actual update procedure is

$$
a_{n+1} =\mathscr{J} \left( \mathcal{N}(a_n + \epsilon g_n) \right)
$$

where the initial input $a_0$ is a scaled random normal distribution. This gradient can be implemented as follows:

```python
def layer_gradient(model, input_tensor, desired_output):
    ...
    input_tensor.requires_grad = True
    output = resnet(input_tensor)
    loss = 0.05*torch.sum(torch.abs(target_tensor - output))
    loss.backward()
    gradient = input_tensor.grad
```

If this method is successful, it would suggest that our model of interest $\theta_1$ has the capability to portray the representation that another model $\theta_2$ has of some output class, and that this portrayal may be made merely by finding the right point in the output space.  And indeed for Class 0 this is the case: observe how the output on the right mimicks GoogleNet's representation of a Tench.

![resnet vectorized to be googlenet tench]({{https://blbadger.github.io}}/neural_networks/resnet_vectorized_to_be_googlenet.png)

The ability of the right point in output space to mimick the representation by another model (for some given class) is even more dramatic when the model's representations of that class are noticeably different.  For example, observe the representation of 'Class 11: Goldfinch' by ResNet and GoogleNet in the images on the left and center below.  ResNet (more accurately) portrays this class using a yellow-and-black color scheme with a dark face whereas GoogleNet's portrayal has reddish feathers and no dark face.  But if we perform the above procedure to ResNet, it too mimicks the GoogleNet output.

![resnet vectorized to be googlenet goldfinch]({{https://blbadger.github.io}}/neural_networks/resnet_vectorized_to_be_googlenet_goldfinch.png)

It may be wondered whether we can apply our gradient descent method to form representations of natural images, rather than generated ones representing some ImageNet category.  All that is required is to change $a_{\theta_2}$ to be some given target image $a_{t}$ before performing the same octave-based gradient descent procedure.  When we choose two images of Dalmatians as our targets, we see that the representations are indeed accurate and that the features they portray are significant: observe how the top image's representation focuses on the body and legs (which are present in the input) whereas the bottom focuses on the face (body and legs not being present in the input).

![resnet vectorized to be googlenet goldfinch]({{https://blbadger.github.io}}/neural_networks/resnet_output_embedding_comparison.png)

### Trivial Autoencoding ability decreases with model depth

It is worth examining again what we have done in the last section: an input $a$ is fed to a model $\theta_1$ to make a target output

$$
\widehat{y} = O(a, \theta_1) 
$$

This target output may be understood as the coordinates in a latent space ($\Bbb R^{1000}$ for ImageNet categories), and the gradient of the distance between the target output $\widehat{y}$ and the actual model's output $y$ is minimized via gradient descent.  In effect the model recieves an input and attempts to copy it using only the knowledge of the coordinates of the chosen latent space, which means that gradient descent on the input to match some latent space coordinates can accurately be described as an autoencoding of the input. 

Autoencoders have long been studied as generative models.  The ability of an autoencoder to capture important features of an input without gaining the ability to exactly copy the input makes these models useful for dimensional reduction as well. And it is this feature that we will explore with our classification model-based autoencoder.

In the mid-2010s, [Goodfellow and colleages](https://arxiv.org/abs/1312.6082) observed empirically that an increase in model depth led to greater generalization, at least for the convolutional neural networks the group was focusing on.  It remains unclear, however, why greater model depth would lead to better generalization.  All else equal, greater model depth requires a larger number of parameters but the same group found that increasing parameter number beyond a certain amount actually decreased generalization, likely due to the propensity for larger models to overfit. 

It has been hypothesized that deep models generalize better because they can be viewed as expressing a program with more steps than shallow models.  But it is unclear why a program with more intructions would generalize better than a program with fewer, indeed the assumption would be that a more specific program would lead to lower generalization via overfitting because the program would be expected to become more specific to the training data set.  If we think about functions instead of programs, a more complicated function is more likely to be able to overfit a general training data set, making it unclear what benefit depth would bring to generalization.

For a typical classification task using deep learning, we seek a model that has approximated some function $p(y \lvert x)$ or else some family of functions out of all possible functions in existence. In the context of feedforward neural networks overfitting may be understood as the result of perfect representation of a model on its input, such that the model has simply approximated a (discontinuous) identity function rather than the function or family of functions desired.  In the feedforward classification setting, an identity function would be the mapping of some input image to a given output such that an arbitrarily small change in the input yields a different output.  In contrast the desired function one wishes to approximate is usually approximately continuous, existing on some manifold.

Perfect representation requires that a model be able to pass all the input information to the model's final layer, or else an identity function would not be approximated as many inputs could give one output.  Common regularizers like weight decay or early stopping act to prevent perfect representation by diminishing a model's capacity to represent the input exactly.

Being that overfitting results from perfect representation and that greater model depth (for cNN-based vision models) prevents overfitting, it may be hypothesized that an increase in model depth prevents perfect representation of the input. 

Are deep learning classification models capable of perfect representation?  This question is somewhat difficult to answer theoretically as it depends on the possible inputs given to a model and the model's configuration, but happily we now have a good way to experimentally test a relationship between model depth and representation capacity. 


The method we will use is the capability of each layer to perform an autoencoding on the input using gradient descent on a random vector as our decoder.  First we pass some target image $a_t$ into a feedforward classification network of choice $\theta$ and find the activations of some given output layer $O_l$

$$
\widehat y = O_l(a_t, \theta)
$$

Now that the vector $\widehat y$ has been found, we want to generate an input that will approximate this vector for layer $l$.  The approximation can be achieved using a variety of different metrics but here we choose $L^1$ for speed and simplicity, making the gradient of interest 

$$
g = \nabla_{a_n} \sum_i \lvert \widehat y - O_l(a_n, \theta) \rvert
$$

The original input $a_0$ is a scaled normal distribution 

$$
a_0 = \mathcal {N}(0.7, 1/20)
$$

and a Gaussian convolution $\mathcal{N_c}$ is applied at each gradient descent to enforce smoothness.

$$
a_{n+1} = \mathcal{N_c} (a_n - \epsilon * g)
$$

Thus the feed-forward network encoder may be viewed as the forward pass to the layer of interest, and the decoder as the back-propegation of the gradient combined with this specific gradient descent procedure.

First we will investigate the ResNet family of models which have the following architecture:

![Resnet layer autoencoding]({{https://blbadger.github.io}}/neural_networks/resnet_architecture.png)

We will first focus on the ResNet 50 model.  Each resnet module has a residual connection, which is formed as follows:

![Resnet layer autoencoding]({{https://blbadger.github.io}}/neural_networks/residual_connection_diagram.png)

Now we will test for each layer's ability to represent the input using our autoencoding method.  We find that early layers are capable of approximately copying the input, but as depth increases this ability diminishes and instead a non-trivial representation is found.

![Resnet layer autoencoding]({{https://blbadger.github.io}}/neural_networks/resnet_autoencoding_perlayer.png)

For each layer, the image formed can be viewed as a result of a trivial (ie approximate copy) and non-trivial (not an approximate copy) representation of the input. In particular, observe the pattern of spots on the dog's fur.  Trivial representations would exactly copy this patter whereas non-trivial ones would not unless this precise patter were necessary for identification (and it is not).  Intuitively a trivial representation would not require any model training to approximately copy the input, as long as the decoder function is sufficiently powerful.  Thus we can search for the presence of trivial representations by repeating the above procedure but for an untrained version of the same model.

![Resnet layer autoencoding]({{https://blbadger.github.io}}/neural_networks/resnet_autoencoding_perlayer2.png)

Thus we see that the untrained (and therefore necessarily trivial) representation of the input disappears in the same deeper layers that the learned (and in this case non-trivial) representation are found for trained models. 

### Imperfect representations are due to nonunique approximation

Why do deep layers of ResNet50 appear to be incapable of forming trivial autoencodings of an input image?  Take layer Conv5 of an untrained ResNet50. This layer has more than 200,000 parameters and therefore viewing this layer as an autoencoder hidden layer $h$ would imply that it is capable of copying the input exactly, as the input has only $299*299=89401$ elements.  

To understand why a late layer in an untrained deep learning model is not capable of very accurate input representation, it is helpful to consider what exactly is happening to make the input representation.  First an input $a$ is introduced and a forward pass generates a tensor $y$ that is the output of a layer of interest in our model,

$$
y = O(a, \theta)
$$

This tensor $y$ may be thought of as containing the representation of input $a$ in the layer in question.  To observe this representation, we then perform gradient descent on an initially randomized input $a_0$ 

$$
a_{n+1} = a_n + \epsilon \nabla_{a_n} O(a_n, \theta)
$$

in order to find a generated input $a_g$ such that some metric $m$ between the output of our original image and this generated image is small.  For an $L^2$ metric, the measure is denoted as

$$
m = ||O(a_g, \theta) - O(a, \theta)||_2
$$

which is the $L^2$ norm of the vector difference between representations $O(a_g, \theta)$ and $O(a, \theta)$.

One can think of each step of the representation visualization process as being composed of two parts: first a forward pass and then a gradient backpropegation step.  An inability to represent an input is likely to be due to one of these two parts, and the gradient backpropegation will be considered first.

It is well known that gradients tend to scale poorly over deeper models with many layers.  The underlying problem is that the process of composing many functions together makes finding an appropriate scale for the constant of gradient update $\epsilon$ very difficult between different layers, and for some instances for multiple elements within one given layer.  Batch normalization was introduced to prevent this gradient scale problem, but although effective in the training of deep learning models it does not appear that batch normalization actually necessarily prevents gradient scaling issues. 

However, if batch normalization is modified such that each layer is determined by variance factor $\gamma = 1$ and mean $\beta = 0$ for a layer output $y$,

$$
y' = \gamma y + \beta
$$

Then the gradient scale problem is averted.  But the initialization process of ResNet50 does indeed set the $\gamma, \beta$ parameters to $1, 0$ respectively, meaning that there is no reason why we would expect to experience problems finding an appropriate $\epsilon$.  Futhermore, general statistical measures of the gradient $\nabla_a O(a, \theta)$ are little changed when comparing deep to shallow layers, suggesting that the gradient used to update $a_n$ is not why the representation is poor.  

Thuse we consider whether the forward pass is somehow the culprit of our poor representation.  We can test this by observing whether the output of our generated image does indeed approximate the tensor $y$ that we attempted to approximate: if so, then the gradient descent process was successful but the forward propegation loses too much information for an accurate representation.

We can test whether the layer output $O(a, \theta)$ approximates $y$ using the $L^2$ metric above, thereby findin the measure of the 'distance' between these two points in the space defined by the number of parameters of the layer in question.  Without knowing the specifics of the values of $\theta$ in all layers leading up to the output layer, however, it is unclear what eactly this metric means, or in other words exactly how small it should be in order to determine if an output really approximates the desired $y$.  

Therefore a comparative metric is introduced: a small (ie visually nearly invisible) shift is made to the original input $a$ and the resulting $L^2$ metric between the output of the original image $y = O(a, \theta)$ and the output of this shifted image

$$
m_r = ||O(a, \theta) - O(a', \theta)||_2
$$

is obtained as a reference.  This reference $m_r$ can then be compared to the metric obtained by comparing the original output to the output obtained by passing in the generated image $a_g$

$$
m_g = ||O(a, \theta) - O(a_g, \theta)||_2
$$

First we obtain $a'$ by adding a scaled random normal distribution to the original,

$$
a' = a + \mathcal N (0, 1/25)
$$

and then we can obtain the metrics $m_r, m_g$ in question.

![Resnet layer distances]({{https://blbadger.github.io}}/neural_networks/layer_distances.png)

Observe that an early layer (Conv2) seems to reflect what one observes visually: the distance between $a', a$ is smaller than the distance between $a_g, a$ as $a'$ is a slightly better approximation of $a$.  On the other hand, the late layer Conv5 exhbits a smaller distance between $a_g, a$ compared to $a', a$ which means that according to the layer in question (and assuming smoothness), there generated input $a_g$ is a better approximation of $a$ than $a'$.

It may be wondered whether this phenomenon is seen for other architectures or is specific to the ResNet50 model.  Typical results for an untrained ResNet18,

![Resnet layer distances]({{https://blbadger.github.io}}/neural_networks/layer_distances_resnet18.png)

and an untrained ResNet152

![Resnet layer distances]({{https://blbadger.github.io}}/neural_networks/layer_distances_resnet152.png)

show that early and late layer representations both make good approximations (relative to a slightly shifted $a'$) of the input they attempt to approximate, even though the late layer representations are visually clearly inaccurate.  Furthermore, observe how the representation becomes progressively poorer at Layer Conv5 as the model exhibits more layers.  These results suggest that in general layer layers of deep learning models are incapable of accurate (trivial) untrained representation of an input not because the gradient backpropegation is necessarily inaccurate but because forward propegation results in a non-unique approximations to the input.

It may be wondered if a better representation method could yield a more exact input. In a later section, we will see that the $a'$ reference is atypically close to $a$ compared to other points in the neighborhood of $O(a', \theta) - O(a, \theta)$ of $O(a, \theta)$ and thus may not be an ideal reference point.  To avoid the issue of what reference in output space to use, we can instead observe the output metric $m_g$ anc compare this to the corresponding metric on the input,

$$
m_i = || a_g - a ||
$$

The $m_g$ metric corresponds to the representation accuracy in the output space and $m_i$ corresponds to the representation accuracy with respect to the input.  Therefore we can think of $m_g$ as being a measure of the ability of the gradient descent procedure to approximate $O(a, \theta)$ while $m_i$ is a measure of the accuracy of the representation to the target input.

For even relatively shallow layers there is an apparent asymptote in $m_i$ far from 0, whereas $m_g$ does tend towards 0.  

![Resnet layer distances]({{https://blbadger.github.io}}/neural_networks/resnet152_conv2_limitations.png)

For deeper layers this effect is more pronounced: it is common for $m_i$ to increase while $m_g$ tends towards the origin.


### Why depth leads to nonunique trivial representations

From the previous few sections, it was seen first that deeper layers are less able to accurately represent an input image than earlier layers for untrained models, and secondly that this poor representation is not due to a failure in the input gradient descent procedure used to visualize the representation but instead results from the layer's inability to distinguish between very different inputs $(a, a_g)$.  It remains to be explored why depth would relate to a reduction in discernment between different inputs.  In this section we explore some contributing factors to this decrease in accuracy from a theoretical point of view before considering their implications to model architectures.

Deep learning vision models are today based on convolutional operations.  To recap exactly what this means...

In the context of image processing, the convolutional operation is non-invertible: given some input tensor $f(x, y)$, the convolution

$$
\omega f(x, y) = f'(x, y)
$$

does not have a unique inverse operation 

$$
\omega^{-1}f'(x, y) = f(x, y)
$$

Specifically, a convolutional operation across an image is non-invertible with respect to the elements that are observed at any given time.  Thus, generally speaking, pixels within the kernal's dimensions may be swapped without changing the ouptut.

Because of this lack of invertibility, there may be many possible inputs for any given output. As the number of convolutional operations increases, the number of possible inputs that generate some output vector also increases exponentially.  Therefore it is little wonder why there are different input $a_g, a, a'$ that all yield a similar output given that many input can be shown to give one single output even with perfect computational accuracy. 

One can experimentally test whether or not non-uniqueness leads to poor representations using simple fully connected architectures.  It should be noted that nearly all fully connected architectures used for classification are composed of non-invertible operations that necessarily lead to non-uniqueness in representation. Specifically, a forward pass from any fully connected layer $x$ to another that is smaller than the previous, $y$, is represented by a non-square matrix multiplication operation. Such matricies are non-invertible, an in particular the case above is expressed with a matrix $A_{m\mathrm{x}n}, m < n$ such that there are an infinite number of linear combinations of elements of $x$.

$$
y = Ax \\
A^{-1}y = x
$$

Whithin a specific range, there are only a finite number of linear combinations but this number increases exponentially upon matrix multiplication composition, where

$$
y = ABCDx \\
$$

This theory is borne out in experimentation, where it appears impossible to make a unique trivial representation of an input of size $a$ with a layer of size $b < a$. In the following figures, the representation visualization method has been modified to have a minimum learning rate of 0.0001 and Gaussian normalization has been removed.  Inputs have been down-scaled to 29x29 for fully connected model feasibility.

![covercomplete depth and representation]({{https://blbadger.github.io}}/neural_networks/under_versus_overcomplete.png)

From the above figure, it may be argued that perhaps the representation generation algorithm is simply not strong enough to capture an accurate representation of the input for smaller layers.  This can be shown to be not the case: plotting the representation accuracies achieved using various iterations of our representation generator, we find that only models with a minimum layer size greater than the input dimension are capable of making arbitrarily accurate representtions.  For the below figure, note that if we continue to trade the 3000-width model, an exponential decay is still observed such that very low (0.1) distance is achieved after 200,000 iterations. 

![iterations and width]({{https://blbadger.github.io}}/neural_networks/width_increased_iterations.png)

On the other hand, the 2000-width model has no decrease in distance from 1,000 to 100,000 iterations even as the embedding distance follows an exponential decay.  These observations provide evidence for the idea that the input representation quality is poor for models with at least one layer with fewer nodes than input elements because of non-uniqueness rather than poor output approximation.

![iterations and width]({{https://blbadger.github.io}}/neural_networks/middle_width_accuracy.png)

But something unexpected is observed for fully connected architectures in which all layers are the same size and identical (or larger than) the input: increased depth still leads to worse representational accuracy.

![covercomplete depth and representation]({{https://blbadger.github.io}}/neural_networks/overcomplete_depth.png)

How could this be?  Each layer is uniquely defined by the last, so non-uniqueness is no longer an issue.  And indeed, if we increase the number of iterations of our gradient descent method for visualization the representation does indeed appear to approximate an input to an arbitrary degree. To be precise, therefore, it is observed for deep models that there are two seemingly contradictory observations: 

$$
m_g = ||O(a, \theta) - O(a_g, \theta)||_2 \\
m_{a'} = ||O(a, \theta) - O(a', \theta)||_2
$$

we can find some input $a_g$ such that

$$
m_{a'} > m_g 
\tag{1}
$$

but for this input $a_g$, 

$$
|| a - a' ||_2 < || a - a_g ||_2
\tag{2}
$$

which is an exact way of saying that the representation is worse even though it makes as good an approximation of $O(a, \theta)$ as $a'$.  

Ignoring biases for now, fully connected linear layers are equivalent to matrix multiplication operations.  If the properties of composed matrix multiplication are considered, we find that there is indeed a sufficient theory as to how this could occur.  Consider first that a composition of matrix multiplications is itself equal to another matrix multiplication,

$$
ABCDx = Ex
$$

Now consider how this multiplication transforms space in $x$ dimensions.  Some basis vectors end up becoming much more compressed or expanded than others upon composition.  Consider the case for two dimensions such that the transformation $Ex$ shrinks $x_1$ by a factor of 1000 but leaves $x_2$ unchanged.  Now consider what happens when we add a vector of some small amount $\epsilon$ to $x$ and find 

$$
|| E(\epsilon) ||
$$ 

the difference of between transformed points $x$ and $x + \epsilon$.  We would end up with a value very near $epsilon_2$.  For example, we could have 

$$ 
\epsilon = 
\begin{bmatrix}
1 \\
1 \\
\end{bmatrix}
$$

But now consider all the possible inputs $a$ that could make 

$$
|| E(a) || \approx || E(\epsilon) ||
$$

If we have the vector

$$ 
a = 
\begin{bmatrix}
1000 \\
0 \\
\end{bmatrix}
$$

For the choice of an $L^1$ metric, clearly

$$
||E(x - a)|| = 1 \approx ||E(x - \epsilon)|| = 1.001
$$ 

even though 

$$
||x - a|| = 1000 > 2 = ||x - \epsilon||
$$  

This example is instructive because it shows us how equations (1) and (2) may be simultanously fulfilled: all we need is a transformation that is contractive much more in some dimensions rather than others.  Most deep learning initializations lead to this phenomenon, meaning that the composition of linear layers gives a transformation that when applied to an n-dimensional ball as an input gives a spiky ball, where the spikes correspond to dimensions that are contracted much more than others.

For an illustration of how this can occur in four dimensions, take an instance where two dimensions denoted in blue that exist on the plane of the page and two more are denoted in red, one of which experiences much more contraction than the other three.  The points that will end up within a certain $L^2$ distance from the target $E(x)$ are denoted for in different dimensions by color.  Two more points are chosen, one in which a small amount $\epsilon$ is added to $a$ and another which could be generated by gradient descent, $a_g$. Observe how the mapping leads to an inversion with respect to the distance between these points and $a$

![spiky ball explanation]({{https://blbadger.github.io}}/neural_networks/spiky_ball_explanation.png)

Why would a representation visualization method using gradient descent tend to find a point like $a_g$ that exists farther from $a$ than $a'$?  We can think of the gradient descent procedure as finding an point $E(a_g)$ as close to $a$ as possible under certain constraints. The larger the difference in basis vector contraction that exists in $E$, the more likely that the point found $E^{-1}(a_g) = a_g$ will be far from $a$.  

As the transformation $E$ is composed of more and more layers, the contraction difference (sometimes called the condition number) between different basis vectors is expected to become larger for most deep learning initialization schemes.  This is important because the input representation method is very similar to the gradient descent procedure of training model parameters, meaning that if poor conditioning leads to a poor input representation then it likely also leads to poor parameter updates for the early layers as well.  

In some respects, $a'$ provides a kind of lower bound to how accurate a point at distance $E(a') - E(a)$ could be.  Observe in the figure above how small a subset of the space around $E(a)$ that $E(a')$ exists inside. Therefore if one were to choose a point at random in the neighborhood of $E(x)$, a point like $E(a')$ satisfying the specific conditions that it does is highly unlikely to be chosen.  

This an be investigated experimentally. Ceasing to ignore biases, we can design a model such that each layer is invertible by making the number of neurons per layer equal to the number of elements in the input.  We design a four-layer network of linear layers only, without any nonlinearities for simplicity.  For each layer, any output $o$ will have a unique corresponding input $x$ that may be calculated by multiplying the output minus the bias vector by the inverse of the weight matrix.

$$
o = Wx + b \implies \\
x = W^{-1}(o-b)
$$

Inverting large matricies requires a number of operations, and it appears that the 32-bit `torch.float` type is insufficient for accurate inversion for the matricies used above.  Instead it is necessary to use `torch.double` type elements, which are 64-bit floating point numbers.  Once this is done, it can be easily checked that the inverse of $O(a, \theta)$ can be found.

With this ability in hand, we can investigate how likely one is to find a point within a certain distance from a target $O(a, \theta)$, denoted $O(a'', \theta)$ such that the input is within some distance of $a$.  We can do this by finding random points near $O(a, \theta)$ by 

$$
O(a'', \theta) = O(a, \theta) + \mathcal{N}(0, 1/1000)
$$

and we can compare the distances between $a''$ and $a$ to the distance between $a'$ and $a$.  The latter is denoted in the upper portion of the following figure, and the distribution of the former in the lower portion.  Observe how nearly all points $a''$ are much farther from $a$ (note the scale: the median distance if more than 40,000) than $a'$ (which is under 3).  This suggests that indeed $a'$ is unusually good at approximating $a$ for points in the neighborhood of $O(a, \theta)$, which is not particularly surprising given that $a'$ was chosen to be a small distance from $a$.

What is more surprising is that we also find that the gradient descent method for visualizing the representation of the output is also far more accurate to the orginal input $a$ than almost all other points in the neighborhood.  In the below example, a short (220 iterations) run of the gradient descent method yields an input $a_g$ such that $m(a, a_g)=2.84$ for an $L^2$ metric but $m(O(a, \theta), O(a_g, \theta)) = 0.52$ with the same metric, which is far larger in output space than the neighborhood explored above but far smaller in input space. Why gradient descent should give such an unusually good approximation of the input for some output neighborhood is currently not clear.

![inverted distances]({{https://blbadger.github.io}}/neural_networks/inverted_distances.png)

How many neurons per layer are required for perfect representation of the input? Most classification models make use of Rectified Linear Units (ReLU), defined as

$$
y = f(x) =
\begin{cases}
0,  & \text{if $x$ $\leq$ 0} \\
x, & \text{if $x$ > 0}
\end{cases}
$$

In this case, the number of neurons per layer required for non-uniqueness is usually much greater than the number of input elements, usually by a factor of around 2.  The exact amount depends on the number of neurons that fulfill the first if condition in the equation above, and if we make the reasonable assumption that $1/2$ of all neurons in a layer do get zeroed out then we would need twice the number of total neurons in that layer compared to input features in order to make an accurate representation.

Input representations require a substantial amount of information from the gradient of the representation with respect to the input in order to make an accurate representation visualization.  This means that one would ideally want to observe $a_g$ after a huge number of gradient descent steps, but for practicality iterations are usually limited to somewhere in the hundreds. 

But curiously enough there is a way to reduce the number of steps necessary: add neurons to the later layers.  Experimentally, increasing the number of neurons in these layers leads to a more accurate representation.  As this cannot result from an increase in information during the forward pass, it instead results from a more accurate gradient passed to the input during backpropegation. 

![middle layer influence]({{https://blbadger.github.io}}/neural_networks/gradient_middle_layer.png)

It is interesting that increasing the number of deep layer neurons is capable of leading to a better input representation for a deep layer even for overcomplete architectures with more layer neurons than input elements. It is probable that increased deep layer neurons prevent scaling problems of gradients within each layer.

Thus we come to the conclusion that the ideal model architecture for trivial representation and thus memorization is the inverse of the architectures commonly used, such that the number of trainable parameters increases at each layer rather than decreases.

### Training does not lead to more accurate approximations in deep layers

What happens to the poor representations in deeper layers upon model training?  We have already seen that training leads to the formation of what was termed a non-trivial representation, ie something that is not simply an approximate copy of the input.  As successful training leads to a decrease in some objective function $J(O(a, \theta)$ such that some desired metric on the output is decreased, it may be hypothesized that training also leads to a decrease in the distance between the representation of the generated input $a_g$ and the representation of the actual input $a$, or more precisely for an $L^2$ distance, the measure decreases toward 0 as the model configuration at the start of training $\theta_0$ is updated during training

$$
||O(a, \theta_n) - O(a_g, \theta_n)||_2 \to 0 \\
\text{as} \; n \to \infty
$$

Intuitively this hypothesis seems reasonable: if a model is trained to recognize images of dalmations as existing in one specific class, it may learn to represent all dalmations in approximately the same way such that a generated image of a dalmatian is in the model's representation more and more similar to any actual dalmatian as training proceeds.  Or put another way, one would expect for a class of images to be represented in approximately the same way such that the distance in that representation for any two inputs decreases during training.

It is somewhat surprising then that this is not the case: the representations for generated versus example dalmatians do not decrease in $L^2$ distance upon a full training run.  Nor does the distance between the original input $a$ and the shifted input $a'$ for trained models in the general case. 

![figure insert]()

The generated input representation $a_g$ does indeed change noticeably during training, but it is clear that this change does not affect the tendancy for deep layers to lack uniqueness in their representations.  Indeed this is clear from the theory expoused in the last section, as the convolutional operation remains non-invertible after training and the spiky ball geometry would not necessarily be expected to disappear as well.

Instead, it appears that during training the possible inputs that make some representation close to the target tensor are re-arranged such that the important pieces of information (in the above case the snout and nose of a dog) are found in the representation, even if the non-uniqueness remains.

### Implications of imperfect input representation

Can deep learning models learn trivial representations regardless of depth?  There is evidence that indeed they can, as observed by [Zhang and colleagues](https://arxiv.org/pdf/1611.03530.pdf): common vision architectures (including GoogleNet) have enough effective capacity to memorize the entire CIFAR10 dataset in which labels were randomly assigned.  But this bears the question: if these models are capable of learning trivial representations, why do they not when they can learn non-trivial ones?  Clearly a non-trivial representation for a model of sufficient depth is in some way more likely to be learned than a trivial one, and indeed Zhang and colleagues observed that models learn non-trivial representations more quickly than trivial ones.

Learning a trivial representation is conceptually similar to the later layers in the model above learning to de-noise the earlier layers.  De-noising autoencoders with parameter higher than a input dimension number are termed overcomplete, and are capable of approximately copying the input upon training.  Being that ResNet and similar models contain far more parameters (>1 million) than inputs (299x299 = 89,401), it is of little surprise that deep learning models are capable of learning to de-noise an input from early layers.

This theory also provides an explanation as to why deep models may prefer to learn non-trivial representations. As the number of possible functions describing the input is smaller in non-trivial versus trivial representation, on average fewer parameters must be adjusted to make an accurate training output.  If a model is of sufficient depth such that either a trivial or non-trivial representation must be learned to lower the objective function, we can expect for a non-trivial one to result if that exists in the training data.  This is likely why non-trivial representations are learned before trivial ones.

The hypothesis that depth prevents overfitting via deeper layers being unable to make trivial representations (without some sort of training) of the input is thus supported.  The observation that even untrained models are capable of copying an input in their early layers but not late layers suggests that the phenonemon is universal to that architecture, and not a result of some specific form of training.

















