## Language Model Features 

### Introduction

How do deep learning models make decisions?  This is a fundamental question for understanding the (often very useful) outputs that these models give.

One way to address this question is to observe that most deep learning models are compositions of linear operations (with nonlinear transformations typically applied element-wise) which means that one can attempt to examine each separable component of a model in relative isolation.  A particularly intuitive question one might ask is what input yields the maximum output for some hidden layer element (assuming other elements of that layer are relatively unchanged) because one may consider that input to be most 'recognizable' to that element.  

Feature visualization studies have shown that vision models learn to identify and generate shapes via a hierarchical sequence of shape speceficity model layer: early convolutional (or transformer) layers are most activated by simple patterns or repeated shapes whereas deeper layers are most activated by specific objects or scenes.  For much more information on this topic, see [this page](https://blbadger.github.io/feature-visualization.html).  

### Background

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

### Llama features are aligned across different-sized models

Before examining which parts of a language model respond to what input, it is helpful to recall what we learned from the same question applied to vision models.  For both [convolutional](https://blbadger.github.io/feature-visualization.html) as well as [transformer](https://blbadger.github.io/transformer-features.html) -based vision models, the main finding was that shallow layers (near the input) learn features that detect simple patterns like repeated lines or checkers, whereas deeper layers learn to identify features that are much more complex (an animal's face, for instance, or a more complicated and irregular pattern).  

We focus here on the Llama family of models introduced by [Touvron and colleages](https://arxiv.org/abs/2302.13971), which are transformer decoder-based causal language models (meaning that these models are trained to predict the next token given a series of previous tokens).

The shallowest Llama modules yield similar results as what was observed for vision models: maximizing the output of any given output neuron for all tokens (here limited to five tokens total) results in a string of identical words. For example, the feature maps of first three neurons of the first Llama transformer (7 billion parameter version) block ($O^l = 1$ if one-indexed)

$$
O_f = [:, \; :, \; 0]  \\
a_g = \mathtt{<unk><unk><unk><unk><unk>}
$$

$$
O_f = [:, \; :, \; 1] \\
a_g = \mathtt{<s><s><s><s><s>}
$$

$$
O_f = [:, \; :, \; 2] \\
a_g = \mathtt{</s></s></s></s></s>}
$$

Here we will use a shorthanded notation for multiple features: $0-4$ for instance indicates features 0, 1, 2, 3, 4 (inclusive) in succession.

$$
O_f = [:, \; :, \; 2000-2004] \\
\mathtt{called \; called \; called \; called \; called} \\
\mathtt{ItemItemItemItemItem} \\
\mathtt{urauraurauraura} \\
\mathtt{vecvecvecvecvec} \\
\mathtt{emeemeemeemeeme}
$$

This is very similar to what is observed for vision transformers: there, the feature maps of single neurons (ie with the same shape as is observed here) in shallow layers yield simple repeated patterns in which the input corresponding to each token is more or less identical ([see here](https://blbadger.github.io/transformer-features.html).

If we observe feature maps from a single neuron at different tokens in the first module of Llama 7b, we find that the word.

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
O_f = [:, \; :, \; 0:4] \\
a_g = </s><unk><s><s><unk>
$$

and four different features combined give

$$
O_f[:, \; :, \; 2000:2004] \\
a_g = \mathtt{vec \; calledura \; calledvec}
$$

This is all to be expected given what was learned from features in vision models. It comes as some surprise therefore to find that different-sized Llama models have nearly identical features for the dimensions they share in common.  For example, if we observe the features of the 13 billion or 30 billion parameter versions of Llama, we find exactly the same features that were present for the 7 billion parameter Llama.  Note that the 13 billion Llama was trained on the same text as the 7 billion Llama, but Llama 30b was trained on more text and thus was exposed to different training data.

```
Transformer Block 1
=================================
Llama 13b [:, :, 2000-2004]
called called called called called
ItemItemItemItemItem
urauraurauraura
vecvecvecvecvec
emeemeemeemeeme

Llama 30b [:, :, 2000-2004]
called called called called called
ItemItemItemItemItem
urauraurauraura
vecvecvecvecvec
emeemeemeemeeme
```

It is worth considering how different this is compared to transformer-based models that are designed for vision tasks. For vision transformers, there is little similarity between identical features for different-sized models. The following figure shows models that were trained on an identical dataset that differ by size only, with each grid location showing the feature map of a certain feature in both models.

![vit feature comparison](https://blbadger.github.io/deep-learning/vit_b_vs_l_features.png)

###  Llama features are aligned across layers

For any given vision transformer neuron, these features are typically very different between different layers. Subsequent layers may have similar features but it is far from the case that every feature in each layer is identifiable from the last layer. In the figure below, it is clear that the features of the first four transformer blocks (where forward- and back-propegation occurs through all previous blocks to the input embedding) are generally distinct.

![vit feature comparison](https://blbadger.github.io/deep-learning/vit_b_32_layers_features.png)

This is in contrast to what is observed for language models, where the features of many outputs are identical. For example, given the 7 billion parameter version of Llama we have the following feature maps at the denoted layer (forward and back-propegation proceeding through all blocks prior)

```
Block 4
[:, :, 2000-2004]
called called called called called
ItemItemItemItemItem
urauraurauraura
vecvecvecvecvec
emeemeemeemeeme

Block 8
called called called called called
ItemItemItemItemItem
urauraurauraura
vecvecvecvecvec
emeemeemeemeeme
```

which is identical to what was seen at block 1.  Deeper layer features are not, however, exactly identical to what is seen for shallower layers: observing the inputs corresponding to may features together at different tokens, we find that an increase in depth leads to substantially different maps for the 13 billion parameter version of Llama.

```
block 1
[:, 0-2, :]
ther� year раreturn
therige year раreturn
ther� year раreturn

block 4
[:, 0-2, :]
tamb would си]{da
column tamb mar marity
cal would tambIONils

block 8
[:, 0-2, :]
ports mar mar user more
endports marре user
tamb marports Elie

block 12
[:, 0-2, :]
ports mar tamb El mar
ports mar cды
tamb marportsiche mar
```

It may also be wondered what the features of each layer (without all preceeding layers) look like.  We see the same identical input representations in most layers of Llama 13b (or Llama 7b or 30b for that matter).  For example, for the 32nd transformer block of this model we see the same features as those we saw for the 1st transformer block.

```
O_f = [:, :, 2000-2003]
called called called called called
ItemItemItemItemItem
urauraurauraura
vecvecvecvecvec
```

This is also not found in vision transformers: although certain features tend to be more or less identical across many transformer blocks the majority are not.

![vit feature comparison](https://blbadger.github.io/deep-learning/vit_b_32_singlelayers_features.png)


### Trained Llama features closely resemble untrained ones

Even more surprising is that an untrained Llama 7b would have identical features to the trained model.

```
Block 1
==================================
O_f = [:, :, 0-2]
<unk><unk><unk><unk><unk>
<s><s><s><s><s>
</s></s></s></s></s>

[:, :, 2000-2004]
called called called called called
ItemItemItemItemItem
urauraurauraura
vecvecvecvecvec
emeemeemeemeeme
```

When a single feature is optimized at different tokens, we find that as for the trained Llama model the token corresponding to the self token

```
[:, 0-2, 2000]
calledremger partlass
are calleddata Alsolass
are I calledamplelass
are Idata calledlass
are Idataample called

Block 4
[:, :, 2000]
are calleddata called met
Item larItemItem met
uraFieldurauraura
vecvecvec reallypha

[:, 0-2, :]
ypearchárodyause
ci helriesiти
haromenIndoz és
```

### Heterogeneous features in deep large Llama

So far we have seen that features are remarkably well-aligned in Llama-based language models.  But when we investigate the larger versions of Llama more closely, we find that 

```
Llama 30b
Block 1
[:, :, 2000-2002]
called called called called called
ItemItemItemItemItem
urauraurauraura

Block 4
[:, :, 2000-2002]
called called called called called
ItemItemItemItemItem
urauraurauraura

Block 32 
[:, :, 2000-2002]
called planlearason current
******** doesn())ason current
ura statlearason current
```

We also find that features of multiple neurons are quite different when comparing Llama 30b to Llama 13b,

```
block 4
Llama 30b
[:, 0-2, :]
osseringering How
oss yearsering yearsoss
estooss yearsoss

Llama 13b
[:, 0-2, :]
tamb would си]{da
column tamb mar marity
cal would tambIONils
```

### Features tested

It may be wondered whether or not the feature generation method employed on this page is at all accurate, or whether the gradient descent-based generated inputs are unlike any real arrangement of tokens a model would actually observe. 

Because we are working with fixed-vocabulary language models, we can go about testing this question by performing a complete search on the input: simply passing each possible input to a model and observing which input yields the largest output for a particular output neuron will suffice.  We will stick with inputs of one token only, as the time necessary to compute model outputs from all possible inputs grows exponentially with input length.




```
Complete search
Trained GPT-2 
[:, :, 0-4]

block 1
Pwr
 Claud
 Peb
 View

block 2
ItemTracker
 watched
 Peb
 (@

block 4
ItemTracker
 watched
 Peb
 (@

block 8
 watched
 Peb
 (@

GPT-2 medium
[:, :, 0-4]
block 1
 Ender
pmwiki
Cass
catentry

block 4
refres
 Flavoring
 carbohyd
assetsadobe

block 8
 refres
 Flavoring
 carbohyd
assetsadobe

block 12
 refres
 Flavoring
 carbohyd
assetsadobe



Untrained GPT-2

block 1
 Yan
 criticizing
policy
 Pont

block 4
inski
 Giants
 consuming
 Bert

block 8
 disruptions
erie
 N
 Aut

block 12
 video
 Collection
 wipe
EngineDebug
```




















