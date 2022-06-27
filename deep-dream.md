## Feature Visualization II: Deep Dream

### Introduction: Feature Optimization

In [Part I](https://blbadger.github.io/feature-visualization.html), deep learning feature maps were investigated by constructing an input that made the highest total activation in that feature, starting with random noise.  But ImageNet does not contain many images that resemble this, making the initial input $a_0$ somewhat artificial.  Instead we can start with a real image for $a_0$ and modify it using gradient descent.

With features from layer Mixed 6d in InceptionV3 maximized by modifying inputs where $a_0$ are selections of flowers, we have

![Dream]({{https://blbadger.github.io}}/neural_networks/InceptionV3_mixed6d_dream.png)

One can also modify the input $a$ such that multiple layers are maximally activated: here we have features from layers 'Mixed 6d' and 'Mixed 6e' jointly optimized, again using a collection of flowers $a_0$.

![Dream]({{https://blbadger.github.io}}/neural_networks/InceptionV3_mixed6d_Mixed6e_dream.png)

### Layer Optimization

It may be wondered what would happen if the input image were optimized without enforcing a prior requiring smoothness in the final image.  Omitting smoothness has a fairly clear justification in this scenario: the starting input is a natural image that contains the smoothness and other statistical features of other natural images, so as long as we do not modify this image too much then we would expect to end with something that resembles a natural image.

This being the case, we can implement gradient descent using octaves without 

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

which is denoted as the sum of the tensor `[:, :, :]` of layer $l$. Using InceptionV3 as our model and applying gradient descent in 3 octaves to an input of flowers, we have

![Dream]({{https://blbadger.github.io}}/neural_networks/dream_mixed6b.png)

Other models yield similar results, with ResNet50

![Dream]({{https://blbadger.github.io}}/neural_networks/Resnet50_layer3_dream.png)

### Image Resolution Flexibility

One interesting attribute of convolutional layers is thay they may be applied to inputs of arbitrary dimension.

As for feature maps, we can apply the gradient descent procedure on inputs of abitrary resolution because each layer is convolutional, meaning that only the weights and biases of kernals are specified such that any size of image may be used as input $a$ without having to change the model parameters $\theta$.  Note that this is only true if all layers up to the final are convolutional: as soon as one desires use of a fully connected architecture, it is usually necessary to use a fixed input size. Right click for images at full resolution.

![Dream]({{https://blbadger.github.io}}/neural_networks/Inception3_dream_mixed6b_layer.png)

![Dream]({{https://blbadger.github.io}}/neural_networks/Inception3_dream_mixed6b_layer_hres2.png)

Note, however, that there is a clear limit to this procedure: although the input image resolution may be increased without bound, the convolutional kernals of the model in question (here InceptionV3) during training were learned only for the resolution that training inputs existed at.  This is why an increase in resolution yields smaller details introduced relative to the whole image, as the changes during gradient descent are made by a model that learned to expect images of a lower resolution.





