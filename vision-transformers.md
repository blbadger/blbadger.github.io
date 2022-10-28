## Vision Transformers

### Introduction

The [convolutional neural network](https://blbadger.github.io/neural-networks.html) has been the mainstay of deep learning vision approaches for decades, dating back to the work of [LeCun and colleagues](https://proceedings.neurips.cc/paper/1989/file/53c3bce66e43be4f209556518c2fcb54-Paper.pdf) in 1989. In the that work, it was proposed that restrictions on model capacity would be necessary to prevent over-parametrized fully connected models from failing to generalize and that these restrictions (translation invariance, local weight sharing etc.) could be encoded into the model itself.

Convolution-based neural networks have since become the predominant deep learning method for image classification and generation due to their parsimonius weight sharing (allowing for larger inputs to be modeled with fewer parameters than traditional fully connected models), their flexibility (as with proper pooling after convolutional layers a single model may be applied to images of many sizes), and above all their efficacy (nearly every state-of-the-art vision model since the early 90s has been based on convolutions).

It is interesting to note therefore that one of the primary motivations of the use of convolutions, that over-parametrized models must be restricted in order to avoid overfitting, has since been found to not apply to deep learning models.  Over-parametrixed fully connected models do not tend to overfit image data even if they are capable of [doing so](https://arxiv.org/abs/1412.6614), and furthermore convolutional models that are currently applied to classify (quite accurately too) large image dataasets are capable of fitting pure noise ([ref](https://dl.acm.org/doi/abs/10.1145/3446776)).

Therefore it is reasonable to hypothesize that the convolutional architecture, although effective and flexible, is by no means required for accurate image classification or other vision tasks. One particularly effective approach has been translated from the field of natural language processing that has been termed the 'transformer'.

We focus on the ViT B 32 model first introduced by [Dosovitsky and colleagues](https://arxiv.org/abs/2010.11929#).  This model is based on the transformer.

![dalmatian vit]({{https://blbadger.github.io}}/neural_networks/vit_dalmatian_representations.png)

![tesla coil vit]({{https://blbadger.github.io}}/neural_networks/vit_representations.png)

