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

Thus we see that the untrained (and therefore necessarily trivial) representation of the input disappears in the same deeper layers that the learned (and in this case non-trivial) representation is found for trained models. 

Can deep learning models learn trivial representations regardless of depth?  There is evidence that indeed they can, as observed by [Zhang and colleagues](https://arxiv.org/pdf/1611.03530.pdf): common vision architectures (including GoogleNet) have enough effective capacity to memorize the entire CIFAR10 dataset in which labels were randomly assigned.  But this bears the question: if these models are capable of learning trivial representations, why do they not when they can learn non-trivial ones?  Clearly a non-trivial representation for a model of sufficient depth is in some way more likely to be learned than a trivial one, and indeed Zhang and colleagues observed that models learn non-trivial representations more quickly than trivial ones.

Learning a trivial representation is conceptually similar to the later layers in the model above learning to de-noise the earlier layers.  De-noising autoencoders with ma parameter dimension higher than an input dimension are termed overcomplete, and are capable of approximately copying the input upon training.  Being that ResNet and similar models contain far more parameters (>1 million) than inputs (299x299 = 89,410), it is of little surprise that deep learning models are capable of learning to de-noise an input from early layers.

This theory also provides an explanation as to why deep models may prefer to learn non-trivial representations. As the number of possible functions describing the input is smaller in non-trivial versus trivial representation, on average fewer parameters must be adjusted to make an accurate training output.  If a model is of sufficient depth such that either a trivial or non-trivial representation must be learned to lower the objective function, we can expect for a non-trivial one to result if that exists in the training data.  This is likely why non-trivial representations are learned before trivial ones.

Our original hypothesis that depth prevents overfitting via deeper layers being unable to make trivial representations (without some sort of training) of the input is thus supported.  The observation that even untrained models are capable of copying an input in their early layers but not late layers suggests that the phenonemon is universal to that architecture, and not a result of some specific form of training.

















