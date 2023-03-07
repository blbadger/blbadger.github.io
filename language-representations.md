## Language Model Representations and Features

### Introduction to Language

One of the most important theorems of machine learning was introduced by [Wolpert](https://direct.mit.edu/neco/article-abstract/8/7/1341/6016/The-Lack-of-A-Priori-Distinctions-Between-Learning) and is colloquially known as the 'No Free Lunch' theorem, which may be stated as the following: no particular machine learning method is better or worse than any other when applied to the problem of modeling all possible statistical distributions.  A similar theorem was shown for the problem of optimization (which covers the processes by which most machine learning algorithms learn) by [Wolpert and Macready](https://ieeexplore.ieee.org/abstract/document/585893), and states that no one algorithm is better or worse than any other across all classes of optimization problems.

These theorems are often used to explain overfitting, in which a model may explain training data arbitrarily well but explains other similar data (usually called validation data) poorly.  But it should be noted that most accurately these theorems do not apply to the typical case in which a finite distribution is being modeled, an d therefore are not applicable to cases of empirical overfitting.  But they do convey the useful idea that each machine learning model and optimization procedure is usueful for only some proper subset of all possible statistical distributions that could exist.

The assumption that not all possible distributions may be modeled by a given learning procedure does not affect the performance of machine learning, if one assumes what is termed the manifold hypothesis.  This hypothesis posits that the set of all elements for a given learning task (ie natural images for a vision model or English sentences for a language model) is a small subset of the set of all elements that could exist in the given space (ie all possible images of a certain resolution or all possible permutations of words and spaces).  The task of the learning program is simply to find the manifold, which is the smooth and approximately connected lower-dimensional space that exists in the higher-dimensional input, such that the objective function associated with the learning task is minimized.

It has been observed on [this page](https://blbadger.github.io/depth-generality.html) that subsequent layers of vision models tend to map arbitrary inputs to manifolds learned during training, rather than simply selecting for certain pieces of input information that may be important for the task at hand. The important difference is that we can see that vision models tend to 'infer' rather than simply select information about their input during the manifold mapping process.  Typically the deeper the layer of the vision model, the more input information is lost in the untrained model and the more information that is inferred in the trained one.

We may wonder whether this is also the case for language models: to these also tend to infer information about their input? Are deeper layers of language models less capable of accurately reconstructing the input, as is the case for vision models?  Do deeper layers infer more input information after training?

### Spatial Learning in Language Model Features

The most prominent deep learning model architecture used to model language is the Transformer, which combines dot product self-attention with feedforward layers applied identically to all elements in an input to yield an output.  Self-attention layers originally were applied to recurrent neural networks in order to prevent the loss of information in earlier words in a sentence, a problem quite different to the ones typically faced by vision models. 

Thus convolutional vision and language models diverged until it was observed that the transformer architecture (with some small modifications) was also effective in the task of image recognition.  Somewhat surprisingly, [elsewhere](https://blbadger.github.io/transformer-features.html) we have seen that transformers designed for vision tasks tend to learn in a somewhat analagous fashion to convolutional models: each neuron in the attention module's MLP output acts similarly to a convolutional kernal, in that the activation of this neuron yields similar feature maps to the activation of all elements in one convolutional filter.

It may be wondered whether transformers designed for language tasks behave similarly.  A first step to answering this question is to observe which elements of an input most activate a given set of neurons in some layer of our language model. One way to test this is by swapping the transformer stack in the [vision transformer](https://arxiv.org/abs/2010.11929) with the stack of a trained language model of identical dimension, and then generating an input (starting from noise) using gradient descent on that input to maximize the output of some element.  For more detail on this procedure, see [this page](https://blbadger.github.io/transformer-features.html).  

To orient ourselves to this model, the following figure is supplied to show how the input is split into patches (which are identical to the tokens use to embed words or bytepairs in language models) via the convolutional stem of Vision Transformer Base 16 to the GPT-2 transformer stack.  GPT-2 is often thought of as a transformer decoder-only architecture, but aside from a few additions such as attention masks the architecture is actually identical to the original transformer encoder and therefore is compatible with the ViT input stem.

![architecture explained]({{https://blbadger.github.io}}/deep-learning/transformer_activation_explained.png)

Here we are starting with pure noise and seek to maximize the activation of a set of neurons in some transformer module using gradient descent between the neurons' value and some large constant tensor.  To be consistent with the procedure used for vision transformers, we also apply Gaussian convolution (blurring) and positional jitter to the input at each gradient descent step (see [this link](https://blbadger.github.io/transformer-features.html) for more details).  First we observe the input resulting from maximizing the activation of an individual neuron (one for each panel, indexed 1 through 16) across all patches. In the context of vision transformers, the activation of each single neuron in all patches (tokens) forms a consistent pattern across the input (especially in the early layers).  

When the activation of a single neuron in all tokens of GPT-2 is maximized using the ViT input convolution with the added transformations of blurring and positional jitter, we have the following:

![gpt2 feature visualization]({{https://blbadger.github.io}}/deep-learning/gpt2_features_viz.png)

The first thing to note is that we see patterns of color forming in a left-to-right and top-to-bottom fashion for all layers.  This is not particularly surprising, given that the GPT-2 language model has been trained to model tokens that are sequential rather than 2-dimensional as for ViTs, and the sequence of tokens for these images is left to right, top to bottom.

It is somewhat surprising, however, that much of the input (particularly for early layers) is unchanged after optimization, implying that as the patch number increases there is a smaller chance that activation of a neuron from that patch actually requires changing the input that corresponds to that patch, rather than modifying earlier patches in the input.  It is not altogether surprising that maximizing the activation of all GPT-2 tokens should lead to larger changes to the first compared to the last tokens.  This is because each token only attends to itself and earlier tokens.  

For an illustration of why we would expect for early input tokens to have more importance than later ones assuming equal importance placed on all inputs, consider the simple case in which there are three tokens output $t_1, t_2, t_3$ and $f'(t_n)$ maximizes the output of all input tokens $i_0, i_1, i_2$ with index $m \leq n$ equally with unit magnitude.  Arranging input tokens in an array as $[i_0, i_1, i_2]$ for clarity,

$$
f'(t_1) = [1, 0, 0] \\
f'(t_1, t_2) = [2, 1, 0] \\
f'(t_1, t_2, t_3) = [3, 2, 1] \\
$$

In the images above we are observing the inputs corresponding to $f'(t_1, t_2, ..., t_n)$.  From the figure above it apears that there is a nonlinear decrease in the input token weight as the token index increases, indicating that there is not in general an equal importance placed on different tokens.  

There also exists a \change in relative importance per starting versus ending token in deeper layers, such that early layer neurons tend to focus on early elements in the input (which could be the first few words in a sentence) whereas the deeper layer neurons focus more broadly.

Positional jitter and Gaussian blurring convolutions are performed on vision models to enforce statistical properties of natural images on the input being generated, namely translation invariance and smoothness.  There is no reason to think that language would have the same properties, and indeed we know that in general language is not translation invariant.

We therefore have motivation to see if the same tendancy to modify the start of the input more than successive layers (as well as more broad pattern of generation with deeper layers) also holds when jitter and blurring are not performed.  As can be seen in the figure below, we see that indeed both observations hold, and that the higher relative importance of starting compared to ending input patches is even more pronounced.

![gpt2 feature visualization]({{https://blbadger.github.io}}/deep-learning/gpt2_features_viz_2.png)

When we instead activate all elements (neurons) from a single patch, we see that in contrast to what is found for vision transformers, early and late layers both focus not only on the self-patch but also preceding ones too.  Only preceding patches are modified because GPT-2 is trained using attention masks to prevent a token peering ahead in the sequence of words.  Note too that once again the deeper layer elements focus more broadly than shallower layer elements as observed above.

![gpt2 feature visualization]({{https://blbadger.github.io}}/deep-learning/gpt2_features_viz_3.png)

The incorperation of more global information in deeper layers is also observed for vision models, although it is interesting to note that transformer-based vision model patches typically do not incorperate as much global information in their deeper layers as MLP-based mixers or convolutional models. 

### Image Reconstruction with Language Models

How much information is contained in each layer of a language model?  One way to get an answer to this question is to attempt to re-create an input, using only the vector corresponding to the layer in question.  Given some input $a$, the output of a model $\theta$ at layer $l$ is denoted $y = O_l(a, \theta)$.  Does $O_l(a, \theta)$ contain the information necessary to re-create $a$? If we were able to invert the forward transformation $O_l$, then it would be simple to recover $a$ by

$$
a = O_l^{-1}(y, \theta)
$$

But typically this is not possible, as for example if multiple inputs $a_1, a_2, ..., a_n$ exist such that

$$
O_l(a_1, \theta) = O_l(a_2, \theta) = \cdots = O_l(a_n, \theta)
$$

which is the case for transformations present in many models used for language and vision modeling. Besides true non-invertibility, linear transformations with eigenvectors of very different magnitudes are often difficult to invert practically even if they are actually invertible.  This is termed approximate non-invertibility, and has been seen to exist for vision models [here](https://blbadger.github.io/depth-generality.html).

The ability of the information present in $O_l$ to generate $a$ from noise can be thought of as a measure of representational accuracy.  How does representational accuracy for transformers trained for language modeling compare to those trained for image classification?  In the following figure, we can see that the trained GPT-2 has less accurate input representations in the early layers than ViT Large does.

![gpt2 vs vit representation]({{https://blbadger.github.io}}/deep-learning/vit_vs_gpt2_representation.png)

This is also the case for out-of-distribution images such as this Tesla coil.  In particular, the first few dozen tokens (top few rows in the grid of the input image, that is) of the input for both images are poorly represented, and display high-frequency inputs that is common for representations of poorly conditioned models. 

![gpt2 vs vit representation]({{https://blbadger.github.io}}/deep-learning/vit_vs_gpt2_representation_2.png)

### Sentence Reconstruction with Language Models

In the previous section we have seen that a trained language model is less capable of representing a visual input than a trained language model (both with similar transformer architectures).  Given the nature of the inputs each model type is trained on, this may not seem very unexpected.  It is more informative to consider the ability of language model layer outputs to reconstruct language inputs, rather than images.

To orient ourselves, first consider the architecture of a typical transformer-based language model.

![gpt2 representation]({{https://blbadger.github.io}}/deep-learning/llm_representation_explained.png)

Language input generation presents a unique challenge to gradient-based methods because language inputs are fundamentally discrete: a word either exists in a certain part of a sentence or it does not.  The standard approach to input generation is to start with a random normal input $a_0 = \mathcal{N}(a, \mu=1/2, \sigma=1/20)$ and then perform gradient descent on some metric (here $L^1$) distance between the target output $O_l(a, \theta)$ for $N$ total iterations, each step being

$$
a_{n+1} = a_n + \eta * \nabla_{a_n} ||O_l(a_n, \theta) - O_l(a, \theta)||_1 \\
\tag{1}\label{eq1}
$$

with $\eta$ decreasing linearly from $\eta$ to $\eta / 10$ as $n \to N$ which empirically results in the fastest optimization.

This method is not useful for language models without modification, given that $\nabla_{a_n}$ is undefined for discrete inputs, which for language models are typically integer tokens.  Instead we must perform gradient descent on some continuous quantity and then convert to and from tokens.  For large language models such as GPT-2, this conversion process occurs using a word-token embedding, which is programmed as a fully connected layer without biases but is equivalent to a (full-rank) matrix multiplication of the input token vector $x$ and the embedding weight matrix $W$ to obtain the embedding vector $e$.

$$
e = Wa
$$

As $e$ is continuous, we can perform gradient descent on this vector such that $e_g$ may be generated from an initially random input $e_0 = \mathcal{N}(e, \mu=1/2, \sigma=1/20)$.

But we then need a way to convert $e_g$ to an input $a_g$. $W$ is usually a non-square matrix given that word encodings often convert inputs with the number of tokens as $n(a) = 50257$ to embeddings of size $n(e) = 768$.  We cannot therefore simply perform a matrix inversion on $W$ to recover $a_g = W^{-1}(e_g)$ because there are fewer output elements than input elements such that there are an infinite number of possible vectors $a_g$ that could yield $e_g$. 

Instead we can use a generalized inverse, also known as the Moore-Pensore pseudo-inverse.  The psuedo-inverse of $W$ is denoted $W^+$, and is defined as

$$
W^+ = \lim_{\alpha \to 0^+} (W^T W + \alpha I)^{-1} W^T
$$

which is the limit from above as $\alpha$ approaches zero of the inverse of $W^T W$ multiplied by $W^T$.  A more understandable definition of $W^+$ for the case where $W$ has many possible inverses (which is the case for our embedding weight matrix or any other transformation with fewer output elements than input elements) is that $W^+$ provides the solution to $y = Wa$, ie $a = W^+y$, such that the $L^2$ norm $\vert \vert a \vert \vert_2$ is minimized.  The pseudo-inverse may be conveniently calculated from the singular value decomposition $W = UDV^T$

$$
W^+ = VD^+U^T
$$

where $D^+$ is simply the transpose of the singular value decomposition diagonal matrix $D$ with all nonzero (diagonal) entries being the reciprocal of the corresponding element in $D$.

Therefore we can instead perform gradient descent on an initially random embedding $e_0 = \mathcal{N}(e, \mu=1/2, \sigma=1/20)$ using

$$
e_{n+1} = e_n + \eta * \nabla_{e_n} ||O_l(e_n, \theta) - O_l(e, \theta)||_1 \\
\tag{2}\label{eq2}
$$

and then recover the generated input $a_g$ from the final embedding $e_g = e_N$ by multiplying this embedding by the pseudo-inverse of the embedding weight matrix $W$,

$$
a_g = W^+e_N
$$

where each token of $a_g$ correspond to the index of maximum activation of this $50257$-dimensional vector.

Thus we can make use of the pseudo-inverse to convert an embedding back into an input token. To see how this can be done with an untrained implementation of GPT-2, the model and tokenizer may be obtained as follows:

```python
import torch
from transformers import GPT2Config, GPT2LMHeadModel

configuration = GPT2Config()
model = GPT2LMHeadModel(configuration)
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'gpt2')
```

Suppose we were given a prompt and a tokenizer to transform this prompt into a tensor corresponding to the tokens of each word. 

```python
prompt = 'The sky is blue.'
tokens = tokenizer.encode(
	  prompt,
	  add_special_tokens=False,
	  return_tensors='pt',
	  max_length = 512,
	  truncation=False,
	  padding=False
	  ).to(device)
```

For GPT-2 and other transformer-based langauge models, the total input embedding fed to the model $e_t$ is the addition of the word input embedding $e$ added to a positional encoding $e_p$

```python
model = model.to(device)
embedding = model.transformer.wte(tokens) 

position_ids = torch.tensor([i for i in range(len(tokens))]).to(device)
positional_embedding = model.transformer.wpe(position_ids)
embedding += positional_embedding
```

The positional weight matrix is invariant for any given input length and thus may be added and subtracted from the input embedding so we do not have to solve for this quantity.  Therefore given the `embedding` variable, we can generate the input tokens by first subtracting the positional embedding $e_p$ from the generated embedding $e_N$ and multiplying the resulting vector by the pseudo-inverse of $W$ as follows:

$$
a_g = \mathrm{arg \; max \;} W^+(e_N - e_p) \\
\tag{3}\label{eq3}
$$

and this may be implemented as

```python
embedding_weight = model.transformer.wte.weight
inverse_embedding = torch.linalg.pinv(embedding_weight)
logits = torch.matmul(embedding - positional_embedding, inverse_embedding)
tokens = torch.argmax(logits, dim=2)[0]
```

It may be verified that Equation \eqref{eq3} is indeed capable of recovering the input token given an embedding by simply encoding any given sentence, converting this encoding to an embedding and then inverting the embedding to recover the input encoding.

Before investigating the representations present in a large and non-invertible model such as GPT-2, we can first observe whether a small and invertible model is capable of accurate input representation (from its output layer). If the gradient descent procedure is sufficiently powerful, we would expect for any input sentence to be able to be generated exactly from pure noise.

The following model takes as input an embedding tensor with `hidden_dim` dimension with the number of tokens being `input_length` and is invertible.  This MLP is similar to the transformer MLP, except without a residual connection and normalization and with equal to or more output elements as there are input elements for each layer (which means that the model is invertible assuming that the GeLU transformation does not zero out many inputs).

```python
class FCNet(nn.Module):

	def __init__(self, input_length=5, hidden_dim=768):
		super().__init__()
		self.input = nn.Linear(input_length * hidden_dim, 4 * input_length * hidden_dim)
		self.h2h = nn.Linear(4 * input_length * hidden_dim, 4 * input_length * hidden_dim)
		self.input_length = input_length
		self.gelu = nn.GELU()
	
	def forward(self, x):
		x = x.flatten()
		x = self.input(x)
		x = self.gelu(x)

		x = self.h2h(x)
		return x
```

Given the target input 'The sky is blue.' it may generated from noise after a few dozen iterations of gradient descent on the output of the model above.  At various $n$ we have the following:

$$
a_{20} = \mathtt{guiActiveUn \; millenniosynウス \; CosponsorsDownloadha} \\
a_{40} = \mathtt{\; this \; millennaウス \;  Cosponsors.} \\
a_{45} = \mathtt{\; this \; millenn \; a \; prompt \; Cosponsors.} \\
a_{50} = \mathtt{ \; this \; is \; a \; prompt  \; sentence.} \\
a_{60} = \mathtt{This \; is \; a \; prompt\; sentence.}
$$

meaning that our small fully connected model has been successfully inverted.

It may be wondered if a non-invertible model (containing one or more layer transformations that are non-invertible) would be capable of exactly representing the input.  After all, the transformer MLP is non-invertible as it is four times smaller than the middle layer. If we change the last layer of our small MLP to have `input_length * hidden_dim` elements, we find that the generated inputs $a_g$ are no longer typically exact copies of the target.

$$
\mathtt{\; this \; millenn \; charismゼウス \; sentence.} \\
\mathtt{\; this \; adolesc \; a \; prompt \; sentence.}
$$

These representations yield the same next character output for a trained GPT-2, indicating that they are considered to be nearly the same as the target input with respect to that model as well.

To get an idea of how effective our gradient procedure is in terms of metric distances, we can construct a shifted input $e'$ as follows

$$
e' = e + \mathcal{N}(e, \mu=0, \sigma=1/20).
$$

Feeding $e'$ into a trained GPT-2 typically results in no change to the decoded GPT-2 output which is an indication that this is an effectively small change on the input. Therefore we can use 

$$
m = || O_l(e', \theta) - O_l(e, \theta)||_1
$$

as an estimate for 'close' to $O_l(e, \theta)$ we should try to make $O_l(e_g, \theta)$. For the first transformer block (followed by the langauge modeling head) of GPT, we can see that after 100 iterations of \eqref{eq2} we have a representation

$$
\mathtt{Engine \; casino \; ozlf \; Territ}
$$

with the distance of output of this representation for the one-block GPT-2 $O_l(e_g, \theta)$ to the target output

$$
m_g = || O_l(e_g, \theta) - O_l(e, \theta)||_1
$$

such that $m_g < m$.

### Language models translate nonsense into sense

One of the primary challenges of large language models today is their ability to generate text that is gramatically and stylistically accurate to the prompt but is inaccurate in some other way, either introducing incorrect information about a topic or else veering off in an unintended direction.  

It can be shown, however, that these models are capable of a much more extreme translation from input nonsense into some real language output by making use the the input representations we have generated in the previous section. Suppose one were given the following prompt: 

$$
\mathtt{The \; sky \; is}
$$

Feeding this input into a trained GPT-2, we get the very reasonable $\mathtt{blue}$ as the predicted next word. This is clearly one of many possible English texts that may yield that same next word to an accurate language model. 

But it can also be shown that one can find many completely nonsensical inputs that also yield the same identical output.  We will see this first with an untrained version of GPT-2 that has been tructated to include a certain number (below only one) of transformer blocks followed by the language modeling head.  The language modeling head allows us to obtain the next predicted word for an input into this model, which provides one measure of 'closeness' if our generated sentence has the same next predicted word as the target input.

```python
class AbbreviatedGPT(nn.Module):

	def __init__(self, model):
		super().__init__()
		self.model = model
	
	def forward(self, x: torch.Tensor):
		# choose the block depth
		for i in range(1):
			x = self.model.transformer.h[i](x)[0]

		x = self.model.lm_head(x)
		return x
```

When we generate an input after a few hundred iterations of Equation \eqref{eq2}, passing in the resulting embeddings to be inverted by Equation \eqref{eq3} for the target input $\mathtt{The \; sky \; is \; blue.}$, we have

$$
 \mathtt{\; Lime \; Lime  \;is \; blueactly} \\
 \mathtt{\; enessidateidate \; postp.}
$$

If a language modeling head is attached to this first transformer block, we find that these two inputs really are viewed nearly equally, in the sense that the next predicted token for both is $\mathtt{"}$ (for one particular random initialization for GPT-2). 

If we increase the number of maximum interations $N$ of our gradient descent procedure in \eqref{eq2} we have 

$$
 \mathtt{\; Terr \; sky \; is \; blue.} \\
 \mathtt{\; cyclists \; sky  \; is \; blue.}
$$

And increasing the total iterations $N$ further (to $N \geq 1000$) yields a smaller $L^2$ distance between $a$ and $a_g$ and a greater probability of recovering the original prompt

$$
\mathtt{The \; sky \; is \; blue.}
$$

although most generated prompts are close but not quite equal to the original. This is unsurprising given that the transformer model is not invertible, such that many inputs may yield an identical output.  

$$
\mathtt{The \; shades \; is \; blue.}
$$

With an increase in the number of transformer blocks before the output modeling head, it becomes more difficult to recover the target inptut $a$.  For example, many iterations of Equation \eqref{eq2} a model with untrained GPT-2 blocks 1 and 2 we have a generated prompt of

$$
\mathtt{The \; sky \; is \; tragedies.}
$$

which is semantically similar (tragedies are sad and the color 'blue' is often colloqially used to mean the same) 

Using the full 12 transformer blocks of an untrained GPT-2, followed by the language modeling head (parameters $N=2000, \eta=0.001$), we can recover inputs that yield the same output character as our original prompt but are completely different.  For example both $a_g$ of

$$
\mathtt{coastline \; DVDs \; isIGHTweak} \\
\mathtt{biologist \; Elephant \; Elephant \; Elephant \; Elephant}
$$

effectively minimize the $L^1$ distance for different initializations of GPT-2, and yield the same next word (bytecode) token as 'The sky is blue.' does.

### Langauge models become less trainable as they are trained

So far we have only considered input representations from untrained models. It may be wondered what the training process does to the model representational ability, and to do so we will use the same abbreviated model configuration above (with GPT-2 transformer blocks following the input and positional embedding, ending in the language modeling head output).

When performing the input representation procedure detailed in the last section on a trained GPT-2 (with a language modeling head), the first thing to note is that the model appears to be very poorly conditioned such that using gradient descent to modify an input to match some output requires careful tuning of $\eta$ and many iterations.  Indeed it takes a truly enormous number of iterations of \eqref{eq2} to generate $e_g$ such that the model's output given $e_g$ is closer to the model's output of $e$ than the slightly shifted input $e'$

 $$
 || O_l(e_g, \theta) - O_l(e, \theta) ||_1 < || O_l(e', \theta) - O_l(e, \theta) ||_1
 \tag{4} \label{eq4}
 $$

on the order to one hundred times as many as for the untrained model to be precise. This very slow minimization of the output loss also occurs when the gradient is calculated on is a different metric, perhaps $L^2$ instead of $L^1$.  A quick check shows that there is no change in the general lack of invertibility of even a single GPT-2 transformer module using this metric.  Thus it is practically infeasible to make the equality in \eqref{eq4} hold if $e'$ is near $e$ (for example, if $e' = e + \mathcal{N}(e, \mu=0, \sigma=1/25$). 

There is a trivial way to satisfy \eqref{eq4} for any $e'$, and this is to simply make the (formerly random) initial embedding $e_0$ to be equal to the target embedding $e$.  Although this is not particularly helpful in understanding the ability of various layers of GPT-2 to represent an input, we can instead make an initial embedding that is a linear combination of random Gaussian noise with the target input.

$$
e_0 = s * \mathcal{N}(e, \mu, \sigma) + t * e
$$

Given an input 

$$
a = \mathtt{This \; is \; a \; prompt \; sentence.}
$$

and with $(s, t) = (0.001, 0.999)$ we have at $N=1000$ an $a_g$ such that \eqref{eq4} holds even for $e'$ very close to $e$.

$$
\mathtt{interstitial \; Skydragon \; a \; behaviゼウス.}
$$

Note, however, that even values $(s, t) = (1, 1)$ give an $e_0$ that is capable of being optimized nearly as well as that above, only that more gradient descent iterations are required.  For example, the following generated input achieves similar output distance to the input above.

$$
\mathtt{opioosponsorsnesotauratedcffff \; conduc}
$$

The inability to effectively modify an input embedding using gradient descent suggests that the gradients obtained from modules of the trained GPT-2 model are inaccurate.  Why would the gradients arriving at the early layers of a model be inaccurate? It may be wondered if this is due to rounding errors in the backpropegation of gradients. One way to check this is to convert both the model and inputs in question to `torch.double()` type, ie 64-bit rather than the default 32-bit floating point values.  Unfortunately there is no significant change in the number of iterations required to make an input that satisfies \eqref{eq4}, and it remains infeasible to satisfy that inequality for $e'$ very close to $e$.

The relative inability of gradient updates to the input embedding to minimize a loss function on the model output suggests that model layers that are adjacent in the backpropegation computational graph (ie the first few transformer encoders) are also poorly optimized towards the end of training.  Indeed, the poor optimization to the input embedding given only one trained transformer block suggests that most of the model is poorly updated towards the end of training, and that is is likely only a few output layers are capable of effective updates at this point.

Is there any way to use gradient descent to more effectively minimize some metric distance between a trained model's output of $a$ versus $a_g$? It turns out that there is: removing the langauge modeling head reduces the number of iterations required to satisfy \eqref{eq4}, by a factor of $>100$ for a one-block GPT-2 model. This makes it feasible to generate $e_g$ that is accurate even when compared with stricter $e'$ but even these embeddings map to inputs that are more or less completely unrecognizable.

$$
\mathtt{srfAttachPsyNetMessage \; Marketable \; srfAttach \; srfAttachPsyNetMessage}
$$

This result indicates that even when capable of minimizing an $L^1$ metric between $O_l(a, \theta)$ and $O_l(a_g, \theta)$, trained language models still cannot differentiate between gibberish and language.

### Approximate Token Mapping

So far we have seen that language model transformer blocks are not invertible and that these models cannot distinguish between gibberish and English language.  It may be wondered if this is due to the discrete nature of the input and language modeling head embeddings: perhaps the $\mathrm{arg \; max}$ of the pseudoinverse of $e_g$ does not find accurate tokens but maybe the second or third highest-activated index could. 

We can select the indicies of the top 5 most activated input token positions as follows:

```python
tokens = torch.topk(logits, 5)[1][0] # indicies of topk of tensor
```

$$
\mathtt{This \; is \; a \; prompt \; sentence.}
$$

For the invertible fully connected network used above one can see that successive outputs are semantically similar, which is perhaps what one would expect given that this model acts on a trained embedding. The top five input token strings are

```
This is a prompt sentence.
 this was an Prompt sentences.(
this are the promptssent,
 This has another speedy paragraph.[
 THIS isna prompted Sent."
```

but for a non-invertible fully connected model (with layer dims of $[768, 4*768, 768]$) we find that only the top one or two most activated input tokens are recognizable.  

```
 this is a prompt sentence.
thisModLoader an Prompt sentences.–
This corridikumanngthlems.<
 THIS are another guiActiveUnDownloadha Mechdragon
 Thisuberty theetsk recomm.(
```

The non-invertible model results above are not wholly surprising given that the model used was not trained such that equivalent inputs would not be expected to be very semantically similar.  But a model composed of a single trained GPT-2 transformer block (no language modeling head) yields only gibberish as well, meaning that training does not confer the desired discriminatory ability on transformer modules.
 
 ```
PsyNetMessage▬▬ MarketablePsyNetMessagePsyNetMessagePsyNetMessage
 srfAttachPsyNetMessagePsyNetMessagequerquequerqueartifacts
ocamp��極artifacts��極 unfocusedRange Marketable
irtualquerqueanwhileizontartifactsquerque
Accessorystaking-+-+ザigslistawaru
```

### Manifold Walk Representations

For the investigation on this page, the main purpose of forming inputs based on the values of the output of some hidden layer is that the input generated gives us an understanding of what kind of information that hidden layer contains.  Given a tensor corresponding to output activations of a layer, the information provided by that tensor is typically invariant to some changes in the input such that one can transform one possible input representation into another without changing the output values significantly. The collection of all such invariants may be thought of as defining the information contained in a layer.

Therefore one can investigate the information contained in some given layer by observing what can be changed in the input without resulting in a significant change in the output, starting with an input given in the data distribution.  The process of moving along in low-dimensional manifold in higher-dimensional space is commonly referred to a 'manifold walk', and we will explore methods that allow one to perform such a walk in the input space, where the lower-dimensional manifold is defined by the (lower-dimensional) hidden layer.

There are a number of approaches to finding which changes in the input are invariant for some model layer output, and of those using the gradient of the output as the source of information there are what we can call direct and indirect methods.  Direct methods use the gradient of some transformation on the output applied to the input directly, whereas indirect methods transform the gradient appropriately and apply the transformed values to the input.

We will consider an indirect method first before proceeding to a direct one.  

Given the gradient of the output with respect to the input, how would one change the input so as to avoid changing the output?  Recall that the gradient of the layer output with respect to the input,

$$
\nabla_a O_l(a, \theta)
$$

expresses the information of the direction (in $a$ space) of greatest increase in $O_l(a, \theta)$ for an infinitesmal change.  We can obtain the gradient of any layer's output with respect to the input by modifying a model to end with that layer before using the following method:

```python
def layer_gradient(model, input_tensor):
	...
	input_tensor.requires_grad = True
	output = a_model(input_tensor)
	loss = torch.sum(output)
	loss.backward()
	gradient = input_tensor.grad
	return gradient, loss.item()
```

Note that only scalars may propegate gradients in the Pytorch autograd engine, meaning that we are actually taking the gradient

$$
\nabla_a \sum_i O_l(a, \theta)_i
$$

but for the purposes on this page, these are effectively equivalent being that no change in the output $O_l(a, \theta)$ also gives us no change in $\sum_i O_l(a, \theta)_i$.

If our purpose is to instead avoid changing the layer's output we want what is essentially the opposite of the gradient, which may be thought of as some direction in $a$-space that we can move such that $O_l(a, \theta)$ is *least* changed.  We can unfortunately not use the opposite of the gradient, as this simply tells us the direction of greatest decrease in $O_l(a, \theta)$.  Instead we want a vector that is orthogonal to the gradient, as by definition an infinitesmal change in a direction (there may be many) that is perpendicular to the gradient does not change the output value.

How can we find an orthogonal vector to the gradient?  In particular, how may we find an orthogonal vector to the gradient, which is typically a non-square tensor?  For a single vector $\mathbf{x}$, we can find an orthogonal vector $\mathbf{y}$ by solving for a solution to the equation of the dot product of these vectors, where the desired product is equal to the zero vector.

$$
\mathbf{x} \cdot \mathbf{y} = 0
$$

We can find that trivially setting $y$ to be the zero vector itself satisfies the equation, and has minimum norm such that simply finding any solution to the above equation is insufficient for our goals. Moreover, language model input are typically matricies composed of many input tokens embedded such that we want to find vectors that are orthogonal to all input token embedding gradients rather than just one.  

To do so, we can make use of the singular value decomposition of the input gradient matrix.  Given a matrix $M$ the singular value decomposition is defined as 

$$
M = U \Sigma V^H
$$

and may be thought of as an extension of the process of eigendecomposition of a matrix into orthonormal bases to matricies that are non-square.  Here the columns of the matrix $U$ is known as the left-singular values of $M$, and the columns of matrix $V$ corresponds to the right-singular values, and $\Sigma$ denotes the singular value matrix that is rectangular diagonal and is analagous to the eigenvalues of an eigendecomposition of a square matrix.  $V^H$ denotes the conjugate transpose of $V$, which for real-valued $V$ is $V^H = V^T$.

The singular value decomposition has a number of applications in linear algebra, but for this page we only need to know that if $M$ is real-valued then $U$ and $V$ are real and orthogonal.  This is useful because for an $M$ that is non-square, we can find the right-singular values $V$ such that $V^H$ is square.  This in turn is useful because some columns (vectors) of $V^H$ are decidedly not orthogonal to $M$ by definition, but as there are more columns in $V^H$ than $M$ we have at least one column that is orthogonal to all columns of $M$. 

Now that a number of orthogonal vectors have been obtained, we can update $e$ to minimize a difference in $O_l(e, \theta)$ by 

$$
e_{n+1} = e_n + \eta * b(V^H_{[j+1]})
$$

where $\eta$ is a sufficiently small update parameter and $j$ indicates the number of tokens in $e$ and $b(V^H)$ indicates a broadcasting of $V^H$ such that the resulting tensor has the same shape as $e_n$, and we repeat for a maximum of $n=N$ iterations.  Alternatively, one can assemble a matrix of multiple column vectors of $V^H$ with the same shape as $e$. Empirically there does not appear to be a difference between these two methods.

For example, let's consider the input

$$
\mathtt{The \; sky \; is \; blue.}
$$

This is a 5-token input for GPT-2, where the embedding corresponding to this input is of dimension $[1, 5, 768]$.  Ignoring the first index (minibatch), we have a $5$ by $768$ matrix.  If we perform unrestricted singular value decomposition on this matrix and recover $V^H$, we have a $[768, 768]$ -dimension orthogonal matrix.  As none of the first $5$ columns of $V^H$ are orthogonal to the columns of $M$ we are therefore guaranteed that the next $763$ columns are orthogonal by definition.

The orthogonal vector approach may therefore be implemented as follows:

```python
def tangent_walk(embedding, steps):
	for i in range(steps):
		embedding = embedding.detach()
		gradient, _ = layer_gradient(a_model, embedding) # defined above
		gradient = torch.squeeze(gradient, dim=0)
		perp_vector = torch.linalg.svd(gradient).Vh[-1] # any index greater than input_length
		embedding = embedding + 0.01*perp_vector # learning rate update via automatically broadcasted vector
	
	return embedding
```

where we can check that the SVD gives us sufficiently orthogonal vectors by multiplying the `perp_vector` by `gradient` via 

```python 
print (gradient @ perp_vector) # check for orthogonality via mat mult
```

If the value returned from this multiplication is sufficiently near zero, the `perp_vector` is sufficiently orthogonal to the `gradient` vector and should not change the value of $\sum O_l(a, \theta)$ given an infinitesmal change in $a$ along this direction.

Unfortunately this method experiences a few challenges and is not capable of finding new locations in $a$-space that do not change $O(a, \theta)$ significantly.  This is due to a number of reasons, the first being that the `perp_vector` is usually not very accurately perpendicular to the `gradient` vector such that multiplication of the orthogonal vector and the gradient returns values on the order of `1e-2` for full transformer architectures.  This is an issue of poor conditioning inherent in the transformer's self-attention module, which can be seen by observing that a model with a single transformer encoder yields values on the order of `1e-3` whereas a three-layer feedforward model simulating the FF present in the transformer module yields values on the order of `1e-8`.

Even when we use a model architecture that allows the SVD to make accurate orthogonal vectors, the instability of the input gradient landscape makes finite learning rates give significant changes in the output, which we do not want. To gain an understanding for what the problem is, suppose one uses the model architecture mimicking the transformer MLP (with three layers, input and output being the embedding dimension of 768 and the hidden layer 4*768).  Obtaining an orthogonal vector from $V^H$ to each token in $e$, we can multiply this vector by $e$ to verify that it is indeed perpendicular.  A typical output for this process is `[ 3.7253e-09, -2.3283e-09,  1.9558e-08, -7.4506e-09,  3.3528e-08]`, indicating that we have indeed found an approximately orthogonal vector.  But when we compare the $L^1$ distance metric on the input $\vert \vert e - e_N \vert \vert_1$ to the same metric on the outputs $\vert \vert O_l(e, \theta) - O_l(e_N, \theta) \vert \vert_1$ we find that the input distance is usually slightly larger than the output distance.  This means that even for a relatively easily-inverted MLP model, the tangent walk approach is insufficient to yield values of $e_N$ that are far from $e$ without changing the output value.

Another approach to changing the input while fixing the output is to clamp portions of the input, add some random amount to those clamped portions, and perform gradient descent on the rest of the input in order to minimize a metric distance between the modified and original output.  The gradient we want may be found via

```python
def target_gradient(model, input_tensor, target_output):
	...
	input_tensor.requires_grad = True
	output = a_model(input_tensor)
	loss = torch.sum(torch.abs(output - target_output))
	print (loss.item())
	loss.backward()
	gradient = input_tensor.grad
	return gradient
```

and the clamping procedure may be implemented by choosing a random index on the embedding dimension and clamping the embedding values of all tokens up to that index, shifting those values by adding these to random normally distributed ones $\mathcal{N}(e; 0, \eta)$.  For a random index $i$ chosen between 0 and the number of elements in the embedding $e$ the update rule $f(e) = e_{n+1}$ may be described by the following equation,

$$
f(e_{[:, \; :i]}) = e_{[:, \; :i]} + \eta * \mathcal{N}(e_{[:, \; :i]}; 0, \eta) \\
f(e_{[:, \; i:]}) = e_{[:, \; i:]} + \epsilon * \nabla_{e_{[:, \; i:]}} || O_l(e, \theta) - O_l(e^*, \theta)  ||_1
$$

where $e^*$ denotes the original embedding for a given input $a$ and $\eta$ and $\epsilon$ are tunable parameters.

```python
def clamped_walk(embedding, steps, rand_eta, lr):
	with torch.no_grad():
		target_output = a_model(embedding)
	embedding = embedding.detach()

	for i in range(steps):
		clamped = torch.randint(768, (1,))
		shape = embedding[:, :, clamped:].shape
		embedding[:, :, clamped:] += rand_eta*torch.randn(shape).to(device)
		gradient = target_gradient(a_model, embedding, target_output)
		embedding = embedding.detach()
		embedding[:, :, :clamped] = embedding[:, :, :clamped] - lr*gradient[:, :, :clamped]
		
	return embedding
```

This technique is far more capable of accomlishing our goal of changing $a$ while leaving $O_l(a, \theta)$ unchanged.  For a 12-block transformer model without a language modeling head such that the output shape is identical to the input shape, tuning the values of $\eta, \epsilon, N$ yields an $L^1$ metric on the distance between $m(e, e_N)$ that is $10$ times larger than $m(O_l(e, \theta), O_l(e_n, \theta))$.  The ratio $r$ defined as

$$
r = \frac{|| e - e_n ||_1} {|| O_l(e, \theta) - O_l(e_N, \theta) ||_1}
$$

may be further increased to nearly $100$ or more by increasing the number of gradient descent iterations per clamp shift step from one to fifty.  

It is interesting to note that the transformer architecture is much more amenable to this gradient clamping optimization than the fully connected model, which generally does not yield an $r>3$ without substantial tuning.

Now that a method of changing the input such that a change in the output is minimized has been found, we can choose appropriate values of $N, \eta, \epsilon$ such that the $e_N$ corresponds to different input tokens.  For successively larger $N$, we have for a trained GPT-2 model (without a language modeling head)

$$
\mathtt{elsius \; sky \; is \; blue.} \\
\mathtt{elsius \; skyelsius \; blue.} \\
\mathtt{elsius \; skyelsiuselsius.}
$$

Once again, we find that the trained GPT-2 approximates an English sentence with gibberish. This is true even if we take the top-k nearest tokens to $e_N$ rather than the top-1 as follows:

```python
elsius skyelsiuselsius.
advertisementelsiusadvertisementascriptadvertisement
eaturesascripteaturesadvertisementelsius
 destroadvertisementhalla blue.]
thelessbiltascript":[{"bilt
```

### Implications

In summary, transformer-based language models such as GPT-2 are unable to distinguish between English sentences and gibberish.

There exists a notable difference between trained language and vision transformer models: the latter contain modules that are at least partially capable of discerning what the input was composed of, whereas the latter does not.  But when we consider the training process for language models, it is perhaps unsurprising that input representations are relatively poor.  Note that each of the gibberish input generations were almost certainly not found in the training dataset precisely because they are very far from any real language.  This means that the language model has no *a priori* reason to differentiate between these inputs and real text, and thus it is not altogether unsurprising that the model's internal representations would be unable to distinguish between the two.

In a sense, it is somewhat more unexpected that language models are so poorly capable of approximate rather than exact input representation.  Consider the case of one untrained transformer encoder detailed in a previous section: this module was unable to exactly represent the input (for the majority of random initializations) but the representations generated are semantically similar to the original prompt.  This is what we saw for [vision models](https://blbadger.github.io/vision-transformers.html), as input representations strongly resembled the target input even if they were not exact copies.  The training process therefore results in the loss of representational accuracy from the transformer encoders.

In this page's introduction, we considered the implications of the 'no free lunch' theorems that state (roughly) that no on particular moel is better than any others at all possible machine learning tasks.  Which tasks a given model performs well or poorly on depends on the model architecture and training protocol, and on this page we saw that trained language models perform quite poorly at the task of input generation because even early layers are unable to differentiate between English sentences and gibberish. Non-invertibility alone does not explain why these trained models are poor discriminators, but the training protocol (simply predicting the next word in a sentence) may do so.

When one compares the difference in vision versus language transformer model input representational ability, it is clear that language models contain much less information on their inputs.  But when we consider the nature of language, this may not be altogether surprising: language places high probability on an extraordinarily small subset of possible inputs relative to natural images.  For example, an image input's identity is invariant to changes in position, orientation (rotation), order, smoothness (ie a grainy image of a toaster is still a toaster), and brightness.  But a language input is not invariant to any of these transformations.  A much smaller subset of all possible inputs (all possible word or bytecode tokens in a sequence) are therefore equivalent to language models, and the invariants to be learned may be more difficult to identify than for vision models.

Finally, it is also apparent that it is much more difficult to optimize an input via gradient descent on the loss of the model output after training versus before.  This appears to be a consistent phenomenon for transformer models as it was also seen in ViTs, but language models experience a greater magnitude of this problem especially when they contain language modeling heads. This observations suggests that efficient training of transformer-based models is quite difficult, and could contribute to the notoriously long training time required for large language models.





