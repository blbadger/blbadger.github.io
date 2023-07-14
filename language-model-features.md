## Language Model Features 

### Introduction

How do deep learning models make decisions?  This is a fundamental question for understanding the (often very useful) outputs that these models give.

One way to address this question is to observe that most deep learning models are compositions of linear operations (with nonlinear transformations typically applied element-wise) which means that one can attempt to examine each separable component of a model in relative isolation.  A particularly intuitive question one might ask is what input yields the maximum output for some hidden layer element (assuming other elements of that layer are relatively unchanged) because one may consider that input to be most 'recognizable' to that element.  

Feature visualization studies have shown that vision models learn to identify and generate shapes via a hierarchical sequence of shape speceficity model layer: early convolutional (or transformer) layers are most activated by simple patterns or repeated shapes whereas deeper layers are most activated by specific objects or scenes.  For much more information on this topic, see [this page](https://blbadger.github.io/feature-visualization.html).  

### Llama features explored

The fundamental challenge of applying input gradient descent or ascent methods is that language model inputs are discrete rather than continuous.  Therefore either some transformation must be made on the input in order to transform it into a continuous and differentiable space or else some other value must be substituted. A logical choice for this other value is to use the embedding of the inputs, as each input has a unique embedding that is continuous and differentiable.  

Therefore we will observe the features present in language models by performing gradient descent on the embedding of the input.

$$
e_{n + 1} = e_n - \eta \nabla_{e_n}|| c - O(e_n, \theta) ||_1
$$

This procedure was considered by previously by [Poerner and colleages](https://aclanthology.org/W18-5437/) but thought to yield generated embeddings $e_g$ that were not sufficiently similar to real word embeddings as judged by cosine distance, which is simply the cosine of the angle between angles of points in space.  For vectors $a$ and $b$

$$
a \cdot b = ||a ||_2 *||b||_2 \cos \theta \\
\cos \theta = \frac{a \cdot b}{|a ||_2 *||b||_2}
$$

It is not clear that this is actually the case for large language models, however. This is because the value of cosine distance is extraordinarily dependent on the number of parameters of $a$ and $b$, with higher-dimensional vectors yielding smaller cosine distances. When $\cos \theta$ of the embedding of a trained Llama 7b is measured between an $e_g$ that most closely matches 'calling' we find that this value is much larger than the cosine distance between embeddings of 'calling' and 'called'.

Secondly, [elsewhere](https://blbadger.github.io/language-discreteness.html) we have already seen that gradient descent on a language model embedding is capable of recovering a text input that exactly matches some given target. If $e_g$ in that case were not sufficiently near real inputs this procedure would have a vanishingly low probability of success.

Block 1
[:, :, 0-4]
<unk><unk><unk><unk><unk>
<s><s><s><s><s>
</s></s></s></s></s>
     
     
[:, :, 0:4]
</s><unk><s><s><unk>
     
[:, :, 400-404]
hththththt
codecodecodecodecode
G G G G G
ateateateateate
essessessessess

[:, :, 400:404]
Gateate G G

[:, 0-4, :]
areremger Alsolass
are Iger Alsolass
arerem mult Alsolass
areremger sonlass
areremger Alsolass

[:, :, 2000-2004]
called called called called called
ItemItemItemItemItem
urauraurauraura
vecvecvecvecvec
emeemeemeemeeme

[:, :, 2000:2004]
vec calledura calledvec

[:, 0, 2000]
called Iger Alsolass
[:, 1, 2000]
are calledger Alsolass
[:, 2, 2000]
are I called Alsolass
[:, 3, 2000]
are Iger calledlass
[:, 4, 2000]
are Iger Also called


block 32?
[:, :, 2000-2001]
called called called called called
ItemItemItemItemItem










