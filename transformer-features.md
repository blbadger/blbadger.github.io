## Transformer and Mixer Features

### Vision Transformer Feature Visualization and Deep Dream

Convolutional models consist of layer of convolutional 'filters', also known as feature maps, that tend to learn to recognize specific patterns in the input (for a detailed look at this phenomenon, see [this page](https://blbadger.github.io/feature-visualization.html)).  These filters are linearly independent from one another at each layer, which makes it perhaps unsurprising that they would select for different possible input characteristics.

At first glance, it would appear that transformers do not contain such easily separated components because although each input is separated into a number of patches that are encoded in a linearly separable manner, the attention transformations act to mix this information, moving relevant information from one token (in this case a patch) to another.  For an interesting look at this in the context of attention-only transformers applied to natural language, see [this work](https://transformer-circuits.pub/2021/framework/index.html) by Elhage and colleagues.  

But we can investigate this assumption by performing a feature visualization procedure similar to that undertaken for convolutional image models on [this page](https://blbadger.github.io/feature-visualization.html).  In brief, we seek to understand how each component of a model responds to the input by finding the input which maximizes the activation of that component, subject to certain constraints on that input in order to make it similar to a natural image.

More precisely, we want to find an input $a'$ such that the activation of our chosen model component $z^l$ is maximized, given the model configuration $\theta$ and input $a$, denoted by $z^l(a, \theta)$ 

$$
a' = \underset{a}{\mathrm{arg \; max}} \; z^l(a, \theta)
$$

Finding the exact value of $a'$ can be very difficult for non-convex functions like hidden layer outputs, so instead we perform gradient descent on an initially random input $a_0$ such that after many steps we have an input $a_g$ that approximates $a'$ in that the activation(s) of component $z_l$ is maximized subject to certain constraints. The gradient here is the tensor of partial derivatives of the $L_1$ metric between a tensor with all values equal to some large constant $C$ and the tensor activation of the element $z^l$

$$
g = \nabla_a (C - z^l_f(a, \theta))
$$

At each step of the gradient descent procedure, the input $a_k$ is updated to $a_{k+1}$ as follows

$$
a_{k+1} = \mathscr J \left( \mathcal N_x(a_k - \epsilon * g_k) \right)
$$

where $\mathcal N$ signifies Gaussian blurring  (to reduce high-frequency input features) and $\mathscr J$ is random positional jitter, which is explained in more detail [here](https://blbadger.github.io/input-generation.html#jitter-using-cropped-octaves).  

For clarity, the following figure shows the shape of the tensors we will reference for this page.  We will focus on the generated inputs that result from maximizing the activation of one or more neurons from the output layer of the MLP layers in various components.  

![vit feature activations]({{https://blbadger.github.io}}/deep-learning/transformer_activation_explained.png)

It is important to note that the transformer's MLP is identically applied across all patches of the input, meaning that it has the same weights and biases no matter where in the image it is applied to.  This is similar to a convolutional operation in which one kernal is scanned across an entire image, except that for the vision transformer the output may be thought of as a stack

![vit feature maps]({{https://blbadger.github.io}}/deep-learning/vit_b_32_feature_map.png)

Maximizing the activation of many neurons  in all patches yields

![vit feature maps]({{https://blbadger.github.io}}/deep-learning/vit_b_32_features_combined.png)

For individual neurons in indivisual patches we have

![vit feature maps]({{https://blbadger.github.io}}/deep-learning/vit_b_32_single_feature.png)

Similar trends are observed for ViT Large 16, which underwent weakly supervised pretraining before ImageNet training

![vit feature maps]({{https://blbadger.github.io}}/deep-learning/vitl16_4_1_16_feature_maps.png)

We can even perform deep dream, as for example using ViT Base 32 we have

![vit dream]({{https://blbadger.github.io}}/deep-learning/vit_b_32_dream.png)

### Spatial Learning in Language Models

So far we have seen that Transformers tend to learn in a somewhat analagous fashion to convolutional models: each neuron in the attention module's MLP output acts similarly to a convolutional kernal, in that the activation of this neuron yields similar feature maps to the activation of all elements in one convolutional filter.

Transformers we first designed to model language, rather than images of the natural world.

![gpt2 feature visualization]({{https://blbadger.github.io}}/deep-learning/gpt2_features_viz.png)

![gpt2 feature visualization]({{https://blbadger.github.io}}/deep-learning/gpt2_features_viz_2.png)

![gpt2 feature visualization]({{https://blbadger.github.io}}/deep-learning/gpt2_features_viz_3.png)
