## Representation in Vision Transformers and Attentionless Models

### Introduction

The [convolutional neural network](https://blbadger.github.io/neural-networks.html) has been the mainstay of deep learning vision approaches for decades, dating back to the work of [LeCun and colleagues](https://proceedings.neurips.cc/paper/1989/file/53c3bce66e43be4f209556518c2fcb54-Paper.pdf) in 1989. In the that work, it was proposed that restrictions on model capacity would be necessary to prevent over-parametrized fully connected models from failing to generalize and that these restrictions (translation invariance, local weight sharing etc.) could be encoded into the model itself.

Convolution-based neural networks have since become the predominant deep learning method for image classification and generation due to their parsimonius weight sharing (allowing for larger inputs to be modeled with fewer parameters than traditional fully connected models), their flexibility (as with proper pooling after convolutional layers a single model may be applied to images of many sizes), and above all their efficacy (nearly every state-of-the-art vision model since the early 90s has been based on convolutions).

It is interesting to note therefore that one of the primary motivations of the use of convolutions, that over-parametrized models must be restricted in order to avoid overfitting, has since been found to not apply to deep learning models.  Over-parametrixed fully connected models do not tend to overfit image data even if they are capable of [doing so](https://arxiv.org/abs/1412.6614), and furthermore convolutional models that are currently applied to classify (quite accurately too) large image dataasets are capable of fitting pure noise ([ref](https://dl.acm.org/doi/abs/10.1145/3446776)).

Therefore it is reasonable to hypothesize that the convolutional architecture, although effective and flexible, is by no means required for accurate image classification or other vision tasks. One particularly effective approach has been translated from the field of natural language processing that has been termed the 'transformer', which makes use of self-attention mechanisms, as well as the mixer architectures that do not make use of attention.

### Transformer architecture

We focus on the ViT B 32 model introduced by [Dosovitsky and colleagues](https://arxiv.org/abs/2010.11929#).  This model is based on the original transformer from [Vaswani and colleages](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html), in which self-attention modules previously applied with recurrent neural networks were instead applied to patched and positionally-encoded sequences in series with simple fully connected architectures.  

The transformer architecture was found to be effective for natural language processing tasks and was subsequently employed in vision tasks after convolutional layers.  But the work introducing the ViT went further and applied the transformer architecture directly to patches of images without any explicit preceding convolutional layers.  It is an open question of how similar these models are to convolutional neural networks.

The transformer is a feedforward neural network model that adapted a concept called 'self-attention' from recurrent neural networks that were developed previously.  Attention modules attempted to overcome the tendancy of recurrent neural networks to pay most attention to sequence elements that directly preceed the current element and forget elements that came before.  The original transformer innovated by applying attention to tokens (usually word embeddings) followed by an MLP, foregoing the time-inefficiencies associated with recurrent neural nets.

In the original self-attention module, each input (usually an embedding of a word) is associated with three vectors $K, Q, V$ for Key, Query, and Value that are produced from learned weight matricies $W^K, W^Q, W^V$.  Similarity between inputs to the first element (denoted by the vector $\pmb{s_1}$) is calculated by finding the dot product (denoted $*$) of one element's query vector with all other element's key vectors 

$$
\pmb{s_1} = (q_1*k_1, q_1*k_2, q_1*k_3,...)
$$

before a linear function is applied to each element followed by a softmax transformation to the vector $\pmb{s_1}$ to make $\pmb{s_1'}$.  Finally each of the resulting scalar components of $s$ are multiplied by the corresponding value vectors for each input $V_1, V_2, V_3,...$ and the resulting vectors are summed up to make the activation vector $\pmb{z_1}$ that is the same dimension as $V_1$

$$
\pmb{s_1'} = \mathbf{softmax} \; ((q_1*k_1)\sqrt d, (q_1*k_2)\sqrt d, (q_1*k_3)\sqrt d,...) \\
\pmb{s_1'} = (s_{11}', s_{12}', s_{13}',...) \\
\pmb{z_1} = V_1 s_{11}' + V_2 s_{12}' + V_3 s_{13}'+ \cdots
$$

The theoretical basis behind the attention module is that certain tokens (originally word embeddings) should 'pay attention' to certain other tokens moreso than average, and that this relationship should be learned directly by the model.  For example, given the sentence 'The dog felt animosity towards the cat, so he behaved poorly towards it' it is clear that the word 'poorly' should be closely associated with the word 'animosity', and the attention module's goal is to model such associations.

But this clean theoretical justification breaks down when one considers that models with such attention modules generally do not perform well on their own but require many attention modules in parallel (termed multi-head attention) and in series.  Given a multi-head attention, one might consider each separate attention value to be context-specific, but it is unclear why then attention should be used at all given that an MLP alone may be thought of as providing context-specific attention.  Transformer-based models are further more many layers deep, and it is unclear what the attention value of an attention value... of a token actually means.

Nevertheless, to gain familiarity with thi model one we note that for multi-head attention, multiple self-attention $z_1$ vectors are obtained (and thus multiple $W^K, W^Q, W^V$ vectors are learned) for each input. The multi-head attention is usually followed by a layer normalization and fully connected layer (followed by another layer normalization) to make one transformer encoder. Attention modules are seriealized by simply stacking multiple encoder modules sequentially.

See [here](https://blbadger.github.io/neural-networks3.html#generalization-and-language-model-application) for an example of a transformer encoder architecture applied to a character sequence classification task.

### Input Generation with Vision Transformers

One way to understand how a model yields its outputs is to observe the inputs that one can generate using the information present in the model itself.  This may be accomplished by picking an output class of interest (here the models have been trained on ImageNet so we can choose any of the 1000 classes present there), assigning a large constant to the output of that class and then performing gradient descent on the input minimizing the difference of the model's output given initial random noise and that large constant, for the specified output index.  

We add two modifications to the technique denoted in the last paragraph: a 3x3 Gaussian convolution is performed on the input after each iteration and after around 200 iterations the input is positionally jittered using cropped regions.  For more information see [this page](https://blbadger.github.io/input-generation.html).  

More precisely, the image generation process is as follows: given a trained model $\theta$ we construct an appropriately sized random input $a_0$

$$
a_0 = \mathcal{N}(a; \mu=0.7, \sigma=1/20)
$$

next we find the gradient of the absolute value of the difference between some large constant $C$ and the output at our desired index $O(a_0, \theta)_i$ with respect to that random input,

$$
g = \nabla_{a_0} |C - O(a_0, \theta)_i|
$$

and finally input is updated by gradient descent followed by Gaussian convolution $\mathcal{N_c}$

$$
a_{n+1} = \mathcal{N_c}(a_n - \epsilon * g)
$$

Positional jitter is applied between updates such that the subset of the input $a_n$ that is fed to the model undergoes gradient descent and Gaussian convolution, while the rest of the input is unchanged.  

$$
a_{n+1[:,m:n,o:p]} = \mathcal{N_c}(a_{n[:,m:n,o:p]} - \epsilon * \nabla_{a_{n[:,m:n,o:p]}} |C - O(a_{n[:,m:n,o:p]}, \theta)_i|)
$$

One of the first differences of note compared to the inputs generated from convolutional models is the lower resolution: of the generated images: this is partly due to the inability of ViT_b_32 to pool outputs before the classification step such that all model inputs must be of dimension $\mathtt{3x224x224}$, whereas most convolutional models allow for inputs to extend to $\mathtt{3x299x299}$ or even beyond $\mathtt{3x500x500}$ due to max pooling layers following convolutions.

When we observe representative images of a subset of ImageNet animal classes,

![vision transformer input generation]({{https://blbadger.github.io}}/neural_networks/vit_animals.png)

as well as landscapes and inanimate objects,

![vision transformer input generation]({{https://blbadger.github.io}}/neural_networks/vit_landscapes.png)

it is clear that recognizable images may be formed using only the information present in the vision transformer architecture just as was accomplished for convolutional models.  

### Vision Transformer hidden layer representations

Another way to understand a model's output is to observe the extent to which various hidden layers in that model are able to autoencode an input: the better the autoencoding, the more complete the information in that layer.

First, let's examine an image that is representative of one of the 1000 categories in the ImageNet 1k dataset: a dalmatian.  We can obtain various layer outputs by subclassing the pytorch `nn.Module` class and accessing the original vision transformer model as `self.model`.  For ViT models, the desired transformer encoder layer outputs may be accessed as follows:

```python
class NewVit(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x: torch.Tensor, layer: int):
        # Reshape and permute the input tensor
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        for i in range(layer):
            x = self.model.encoder.layers[i](x)

        return x
```
where `layer` indicates the layer of the output desired (`self.model.encoder.layers` are 0-indexed so the numbering is accurate). The `NewVit` class can then be instantiated as

```python
vision_transformer = torchvision.models.vit_b_32(weights='IMAGENET1K_V1')
new_vision = NewVit(vision_transformer).to(device)
new_vision.eval()
```

First let's observe the effect of layer depth (specifically transformer encoder layer depth) on the representation accuracy of an untrained ViT_B_32 and compare this to what was observed for ResNet50 (which appears to be a fairly good stand-in for other convolutional models).


![dalmatian vit]({{https://blbadger.github.io}}/neural_networks/vit_vs_resnet_untrained_representations.png)

The first notable aspect of input representation is that it appears to be much more difficult to approximate a natural image using Gaussian-convolved representations for ViT than for ResNet50, or more precisely it is difficult to find a learning rate $\epsilon$ such that the norm of the difference between the target output $O(a, \theta)$ and the output of the generated input $O(a_g, \theta)$ is smaller than the norm of the difference between the target output and the output of a slightly shifted input $O(a', \theta)$ where $a' = a + \mathcal{N}(a; \mu=0, \sigma=1/18)$, meaning that it is difficult to obtain the following inequality:

$$
||O(a, \theta) - O(a', \theta)||_2 < ||O(a, \theta) - O(a_g, \theta)||_2
$$


![dalmatian vit]({{https://blbadger.github.io}}/neural_networks/vit_dalmatian_representations.png)


![tesla coil vit]({{https://blbadger.github.io}}/neural_networks/vit_representations.png)


### Vision Transformer Deep Dream

### Patch-based models without attention







