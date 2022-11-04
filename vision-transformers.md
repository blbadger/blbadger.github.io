## Representation in Vision Transformers and Attentionless Models

### Introduction

The [convolutional neural network](https://blbadger.github.io/neural-networks.html) has been the mainstay of deep learning vision approaches for decades, dating back to the work of [LeCun and colleagues](https://proceedings.neurips.cc/paper/1989/file/53c3bce66e43be4f209556518c2fcb54-Paper.pdf) in 1989. In the that work, it was proposed that restrictions on model capacity would be necessary to prevent over-parametrized fully connected models from failing to generalize and that these restrictions (translation invariance, local weight sharing etc.) could be encoded into the model itself.

Convolution-based neural networks have since become the predominant deep learning method for image classification and generation due to their parsimonius weight sharing (allowing for larger inputs to be modeled with fewer parameters than traditional fully connected models), their flexibility (as with proper pooling after convolutional layers a single model may be applied to images of many sizes), and above all their efficacy (nearly every state-of-the-art vision model since the early 90s has been based on convolutions).

It is interesting to note therefore that one of the primary motivations of the use of convolutions, that over-parametrized models must be restricted in order to avoid overfitting, has since been found to not apply to deep learning models.  Over-parametrixed fully connected models do not tend to overfit image data even if they are capable of [doing so](https://arxiv.org/abs/1412.6614), and furthermore convolutional models that are currently applied to classify (quite accurately too) large image dataasets are capable of fitting pure noise ([ref](https://dl.acm.org/doi/abs/10.1145/3446776)).

Therefore it is reasonable to hypothesize that the convolutional architecture, although effective and flexible, is by no means required for accurate image classification or other vision tasks. One particularly effective approach has been translated from the field of natural language processing that has been termed the 'transformer', which makes use of self-attention mechanisms, as well as the mixer architectures that do not make use of attention.

### Transformer architecture

We focus on the ViT B 32 model introduced by [Dosovitsky and colleagues](https://arxiv.org/abs/2010.11929#).  This model is based on the original transformer from [Vaswani and colleages](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html), in which self-attention modules previously applied with recurrent neural networks were instead applied to patched and positionally-encoded sequences in series with simple fully connected architectures.  

The transformer architecture was found to be effective for natural language processing tasks and was subsequently employed in vision tasks after convolutional layers.  But the work introducing the ViT went further and applied the transformer architecture directly to patches of images without any explicit preceding convolutional layers.  It is an open question of how similar these models are to convolutional neural networks.

### Input Generation with Vision Transformers

One way to understand how a model yields its outputs is to observe the inputs that one can generate using the information present in the model itself.

### Vision Transformer hidden layer representations

Another way to understand a model's output is to observe the extent to which various hidden layers in that model are able to autoencode an input: the better the autoencoding, the more complete the information in that layer.

![dalmatian vit]({{https://blbadger.github.io}}/neural_networks/vit_dalmatian_representations.png)

![tesla coil vit]({{https://blbadger.github.io}}/neural_networks/vit_representations.png)

### Patch-based models without attention





