## Language Model Features 

### Introduction

How do deep learning models make decisions?  This is a fundamental question for understanding the (often very useful) outputs that these models give.

One way to address this question is to observe that most deep learning models are compositions of linear operations (with nonlinear transformations typically applied element-wise) which means that one can attempt to examine each separable component of a model in relative isolation.  A particularly intuitive question one might ask is what input yields the maximum output for some hidden layer element (assuming other elements of that layer are relatively unchanged) because one may consider that input to be most 'recognizable' to that element.  

Feature visualization studies have shown that vision models learn to identify and generate shapes via a hierarchical sequence of shape speceficity model layer: early convolutional (or transformer) layers are most activated by simple patterns or repeated shapes whereas deeper layers are most activated by specific objects or scenes.  For much more information on this topic, see [this page](https://blbadger.github.io/feature-visualization.html).  

### Introduction

The fundamental challenge of applying input gradient descent or ascent methods is that language model inputs are discrete rather than continuous.  Therefore either some transformation must be made on the input in order to transform it into a continuous and differentiable space or else some other value must be substituted. A logical choice for this other value is to use the embedding of the inputs, as each input has a unique embedding that is continuous and differentiable.  

Therefore we will observe the features present in language models by performing gradient descent on the embedding of the input.

$$
e_{n + 1} = e_n - \eta \nabla_{e_n}|| c - O(e_n, \theta) ||_1
$$

This procedure was considered by previously by [Poerner and colleages](https://aclanthology.org/W18-5437/) but thought to yield generated embeddings $e_g$ that were not sufficiently similar to real word embeddings as judged by cosine distance, which is simply the cosine of the angle between angles of points in space.  For vectors $a$ and $b$

$$
a \cdot b = ||a ||_2 *||b||_2 \cos \phi \\
\cos \phi = \frac{a \cdot b}{|a ||_2 *||b||_2}
$$

It is not clear that this is actually the case for large language models, however. This is because the value of cosine distance is extraordinarily dependent on the number of parameters of $a$ and $b$, with higher-dimensional vectors yielding smaller cosine distances. When $\cos \theta$ of the embedding of a trained Llama 7b is measured between an $e_g$ that most closely matches 'calling' we find that this value is much larger than the cosine distance between embeddings of 'calling' and 'called'.

Secondly, [elsewhere](https://blbadger.github.io/language-discreteness.html) we have already seen that gradient descent on a language model embedding is capable of recovering a text input that exactly matches some given target. If $e_g$ in that case did not sufficiently resemble real inputs this procedure would have a vanishingly low probability of success.

With this in mind, we can go about observing what inputs activate each neuron in various layers of language models.  We will consider the outputs of transformer-based language models that are sometimes described as having a shape $[batch, \; sequence, \; feature]$.  We will generate only 1-element batches, so on this page any output of shape $[:, n, m]$ is equivalent to the ouptut $[0, n, m]$ where $[:]$ indicates all of the elements of a given tensor index.

The following figure provides more detail into what exactly we are considering a 'feature'.  Each 'feature' is essentially a single neuron's activation of the output of a transformer block (although it could also be assigned as the output of a transformer's self-attention module as this is the same shape).  

![llm_features](https://blbadger.github.io/deep-learning/llm_features_explained.png)

To find an input $a_g$ that represents each 'feature' of output layer $O^l$, we want to find an input $a_g$ that maximized the activation of some subset of that output layer for the model $\theta,

$$
a_g = \underset{a}{\mathrm{arg \; max}} \; O^l_f(a, \theta)
$$

with the reasoning that the input $a_g$ gives a large output in the feature $O^l_f$ and therefore is most emblematic of the sort of input that this feature recognizes in real inputs. 

Language inputs are fundamentally discrete, and so we cannot simply perform gradient descent on the input as was done for vision models. We can, however, perform gradient descent on the embedding of an input, which is defined as the matrix multiplication of the input $a$ and an embedding weight $e$. For language models the input is typically a series of integers which is converted into a vector space embedding via a specialized weight matrix $W$.  We will denote this mapping as follows:

$$
e = Wa
$$

and ignore the fact that $a$ cannot be multiplied directly to a weight matrix as it is not a vector for now. 

We can use gradient descent on the input's embedding where the objective function is a distance metric between a tensor of the same shape as $O^l_f$ comprised of some large constant $C$ and the output values as follows.  Here we choose an $L^1$ metric on the output.

$$
e_{n+1} = e_n + \eta \nabla_{e_n} (||C - O^l_f(e_n, \theta)||_1)
$$

After $N$ iterations we generate $e_g$, and recover the input which maps to this embedding using the Moore-Penrose pseudoinverse

$$
a_g = W^+e_g
$$

where $a_g$ is a vector that can be converted to a series of tokens by taking the maximum value of each sequence element in $a_g$.

### Llama features are nearly identical between models

Before examining which parts of a language model respond to what input, it is helpful to recall what we learned from the same question applied to vision models.  For both [convolutional](https://blbadger.github.io/feature-visualization.html) as well as [transformer](https://blbadger.github.io/transformer-features.html) -based vision models, the main finding was that shallow layers (near the input) learn features that detect simple patterns like repeated lines or checkers, whereas deeper layers learn to identify features that are much more complex (an animal's face, for instance, or a more complicated and irregular pattern).  

The first modules of language models appear to give a similar result: maximizing the output of any given output neuron for all tokens (here limited to five tokens total) results in a string of identical words. For example, the output of the first transformer block ($O^l = 1$ if one-indexed)

$$
O_f = [:, :, 0] = \\
a_g = \mathtt{<unk><unk><unk><unk><unk>}
$$

$$
O_f = [:, :, 1] \\
a_g = \mathtt{<s><s><s><s><s>}
$$

$$
O_f = [:, :, 2] \\
a_g = \mathtt{</s></s></s></s></s>}
$$

on this page, we will use a shorthanded notation for multiple features: $0-4$ for instance indicates features 0, 1, 2, 3, 4 (inclusive) in succession.

$$
O_f = [:, :, 2000-2004] \\
\mathtt{called \; called \; called \; called \; called} \\
\mathtt{ItemItemItemItemItem} \\
\mathtt{urauraurauraura} \\
\mathtt{vecvecvecvecvec} \\
\mathtt{emeemeemeemeeme}
$$

$$
O_f = [:, 0 - 4, 2000] \\
\mathtt{\color{red}{called}\; Iger \; Alsolass} \\
\mathtt{are \; \color{red}{called}ger \; Alsolass} \\
\mathtt{are \; I \; \color{red}{called} \; Alsolass} \\
\mathtt{are \; Iger \; \color{red}{called}lass} \\
\mathtt{are \; Iger \; Also \; \color{red}{called}} \\
$$

When we combine features, somewhat unpredictable outputs are formed.  For example, optimizing an input for the first four features (denoted `0:4`, note that this is non-inclusive) yields

$$
O_f = [:, :, 0:4]
a_g = </s><unk><s><s><unk>
$$

and four different features combined give

$$
O_f[:, :, 2000:2004]
a_g = vec calledura calledvec
$$



Llama 13b
Block 1
[:, :, 2000-2004]
called called called called called
ItemItemItemItemItem
urauraurauraura
vecvecvecvecvec
emeemeemeemeeme

Llama 30b
[:, :, 2000-2004]
called called called called called
ItemItemItemItemItem
urauraurauraura
vecvecvecvecvec
emeemeemeemeeme

It is worth considering how different this is compared to transformer-based models that are designed for vision tasks. 

![vit feature comparison](https://blbadger.github.io/deep-learning/vit_b_vs_l_features.png)

###  Llama features are aligned across layers

For any given transformer neuron, these features are typically very different between different layers, such that for vision transformers it is not usually possible to tell which feature map corresponds to which neuron given feature maps from the previous layer.



llama 7b 
block 32
[:, :, 2000-2001]
called called called called called
ItemItemItemItemItem

llama 13b

Block 1
[:, :, 2000-2004]
called called called called called
ItemItemItemItemItem
urauraurauraura
vecvecvecvecvec
emeemeemeemeeme

Block 1-4
[:, :, 0]
<unk><unk><unk><unk><unk>

Block 1-8
[:, :, 0]
<unk><unk><unk><unk><unk>

Block 1-4 
[:, 0, 2000]
called� year раreturn
[:, 1, 2000]
ther called year раreturn
[:, 2, 2000]
ther� called раreturn

Block 31
[:, :, 2000] top5
called called called called called
ther Consultado Webertreturn
itoные yearION Port
childrenvecAF Grerr
display connectdbhing job

block 1
[:, :, 2000] top5
called called called called called
ther� year раreturn
itovec Web ·ATE
         ConsultadoAFhingја
childrenpacedbert Port

block 1 
[:, :, 0-4]
<unk><unk><unk><unk><unk>
<s><s><s><s><s>
</s></s></s></s></s>
�����
     
block 1
[:, 0-2, :]
ther� year раreturn
therige year раreturn
ther� year раreturn

block 1-4
[:, 0-2, :]
tamb would си]{da
column tamb mar marity
cal would tambIONils


block 1-8
[:, 0-2, :]
ports mar mar user more
endports marре user
tamb marports Elie

block 1-12
[:, 0, :]
ports mar tamb El mar
[:, 1, :]
ports mar cды
[:, 2, :]
tamb marportsiche mar

It is important to note that this alignment of features across many layers of a transformer is not what was observed for vision models.  In that case, subsequent layers may have similar 

![vit feature comparison](https://blbadger.github.io/deep-learning/vit_b_32_layers_features.png)

![vit feature comparison](https://blbadger.github.io/deep-learning/vit_b_32_singlelayers_features.png)

### Heterogeneous features in deep large Llama

Llama 30b
Block 1
[:, :, 0-4]
<unk><unk><unk><unk><unk>
<s><s><s><s><s>
</s></s></s></s></s>
     

[:, 0-4, :]
********UNlearason current
******** statlearason current
********UNlearason current
********UN suason OP
********UN suason current

[:, :, 0:4]
<unk><s><s></s> 

[:, :, 400-401]
hththththt
codecodecodecodecode


[:, :, 2000-2004]
called called called called called
ItemItemItemItemItem
urauraurauraura
vecvecvecvecvec
emeemeemeemeeme

[:, 0, 2000]
called statlearason current
[:, 1, 2000]
******** calledlearason current
[:, 2, 2000]
******** stat calledason current
[:, 3, 2000]
******** statlear called current
[:, 3, 2001]
******** statlearItem current

Blocks 1-4
[:, :, 2000-2004]
called called called called called
ItemItemItemItemItem
urauraurauraura

blocks 1-4
[:, 0-2, :]
osseringering How
oss yearsering yearsoss
estooss yearsoss

blocks 1-32 (50 iterations)
[:, :, 0-2]
<unk> implement<unk><unk><unk>
</s>big</s></s></s>

[:, :, 2000-2003]
called planlearason current
******** doesn())ason current
ura statlearason current
vecologicalvecvecvec

block 32
[:, :, 2000-2003]
called called called called called
ItemItemItemItemItem
urauraurauraura
vecvecvecvecvec










