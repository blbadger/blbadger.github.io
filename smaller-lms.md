## Smaller, Better Language Models with Masked Mixers

### Background

The training of the most effective language models today (3/2024) requires an enormous amount of computational resources: a whopping 1720320 hours of 80GB nvidia A100 compute time were required to train the 70 billion parameter version of [Llama 2](https://arxiv.org/pdf/2307.09288.pdf). Assuming that the meta RSC was used (6080 GPUs), this comes out to nearly two weeks of nonstop training for that entire cluster.  

This prohibitive amount of compute power required is mostly down to the very large size of the models that are currently trained, not the training dataset itself: most LLMs are trained on between 1 and 5 trillion tokens of text, but this is not actually that much information when one considers that each token is typically expressed as bytecode (ie one byte) and therefore the training dataset is a few terabytes, substantially smaller than the large image datasets required to train (much smaller) diffusion models.

Current state-of-the-art transformer models are very large indeed (~100 billion parameters) but with transformers there is another wrinkle: the key, query, and value projection weights form gradients for all of a training input's sequence.  This means that a 7 billion parameter Llama model will actually require much more space than the 7*=14 GB one might think (if training in 16 bit precision, ignoring optimizers for the moment) for any reasonable context length, and thus has an 'effective' parameter size much larger than the model's actual size during inference. Together this means that it takes hundreds of gigabytes of vRAM to train a model on a small context window and with a small batch size, even though this model can fit in 14 gigabytes of memory during inference.

This begs the question: are these actual and 'effective' parameters necessary? It is clear that transformer-based models do indeed become substantially more effective with an increase in the number of parameters they contain, but if we were not restricted to one particular architecture is it possible that we could design a model with far fewer parameters for language modeling?

A very quick calculation suggests that billions or even millions of parameters are far more than would be necessary to model the English language. It has been claimed that there are somewhere around $10^{570}$ possible English sentences, as an upper bound. Without knowing how to model these sentences, we can view them as unique points in an arbitrarily high-dimension space. Now due to a theorem of high-dimensional space, the same points may be obtained with arbitrary precision in a space that is approximately $\log 10^{570} = 570$ dimensional.  This means that a model with the same number of parameters may exist such that each sentence may be found via a combination of 570 parameters, much less than the billions of parameters typically used for language models today. 

That being the case, one can assume that far fewer parameters are required to model language than are currently found in LLMs but it cannot be assumed that a model with that number of parameters is actually trainable: it could be that training requires a large model that must then be converted into a small model.  This is the approach used when performing pruning, where parameters are dropped depending on their importance for some output. Alternatively, instead of removing parameters one could reduce the memory required to store each parameters: this is the approach of quantization methods, which are perhaps the most effective methods currently available for shrinking the effective size of a model. 

The observation that weight quantization rather than pruning is the most effective method for reducing a transformer model's effective size suggests that this architecture may indeed require nearly all the trained parameters in order to function effectively, although whether this is the case or not remains an open questions. Regardless, we take the different approach of investigating new architectures and training new models on this page.

### Introduction

The goal of any deep learning architectural choice is fundamentally one of efficiency: as it has long been known that even a simple one-hidden-layer fully connected nerual network is capable of approximating any arbitrary function, although not necessarily capable of *learning* to approximate any function.  Empirical observation over the last decade suggests that indeed if one is given enough compute power, most architectural choices may result in the same quality of output if sufficient numerical stabilization and dataset size are used (and the model follows some general rules, such as exhibiting sufficient depth).  If one were given an unlimited computing budget with unlimited data, then, a model's architecture is unimportant.

But for real-world scenarios where compute is limited, the choice of a model's architecture becomes very important. An example of this is the introduction of exclusively transformer-based architectures to image categorization, where these models were able to achieve performance on par with convolutional models but required vastly more compute power to do so and were thus of dubious practical significance for that task (although they are indeed practically useful for generative image modeling).

With this in mind, what we will seek is a model architecture that is more efficient than the transformer with respect to training: given some fixed language dataset and compute budget, we want a model that is more effective (ie reaches a lower validation loss and generates more cohesive language) than a transformer-based state-of-the-art model.

In scientific endeavors it is often useful to begin with small-scale experiments when testing basic ideas, before moving to larger-scale ones for further experimentation. In this spirit (and because of the author's limited compute) we will test small language models (millions rather than billions of parameters) on small datasets (megabytes rather than terabytes).

The experimental setup will be as follows: for our dataset we will start with TinyStories, which is a collection of simple text that contains a limited vocabulary similar to what a four-year-old would use that allows for effective modeling by transformers in the millions rather than billions of parameter scale. For reference, small versions of state-of-the-art transformer models will be trained and used as a comparison to the new architectures that we will try here.

### Language Mixer Basics

[Elsewhere](https://blbadger.github.io/language-discreteness.html) it was observed that transformers exhibit a somewhat unexpected phenomena: firstly that transformer blocks must be extremely wide (embedding size $e > 3000$) in order to have any accurate input representation ability, and secondly that the ability of a transformer to accurately represent a token it has seen previously (ie a 'non-self' token) disappears after training.  On the other hand, a modification of the MLP mixer architecture was found to have accurate self- token representation even from very small models with $e < 100$ (non-self representation is only accurate if expansions are not used in the convolutional layers, see below for more information). Thus this may be a good candidate with which to start the process of looking for more effective architectures than the transformer, an architecture that although effective has inefficiencies acknowledged by the authors of the original 'Attention is all you need' paper (GTC 2024).

The MLP mixer architecture is conceptually similar to a transformer if all the multi-head attention layers were replaced with linear transformations over the sequence dimension. This was originally designed for vision tasks, and we will employ a modification of this architecture for language.  The choice of this starting point is mostly due to simplicity (this model does not require positional encoding or attention) and representational efficiency (even small models can accurately represent non-self tokens).

The mixer has previously been applied only to vision modeling, where it was found to be not quite as efficient as a transformer of equivalent 'size' for a fixed dataset (the only instance of a mixer applied to language modeling tasks is a nanoscale truncated mixer applied to bloom filtered text that has obvious unsuitibilities as a generative model).  It is important to observe, however, that with langauge one is typically not bound by a dataset's size but rather the amount of compute one can bring to that dataset. This suggests that a mixer architecture might be a good place

![mixer]({{https://blbadger.github.io}}deep-learning/llm_mixer.png)

First, we define the operations on one mixer block, which is a module akin to one transformer block, except without self-attention. 

```python
class MixerBlock(nn.Module):

	def __init__(self, dim, length, mixer_mask=True, expand_conv=True):
		super().__init__()
		self.patch_layernorm = nn.LayerNorm(dim)
		self.seq_layernorm = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim)
		self.expand_conv = expand_conv
		if self.expand_conv:
			self.conv = ConvForward(length)
		else:
			self.conv = nn.Conv1d(length, length, 1)
		
		# for CLM training, apply lower triangular mask to convolution weights during the forward pass
		self.mixer_mask = mixer_mask
```

The last line of the code block above passes the boolean `mixer_mask` to a class variable to be accessed during the forward pass of the `MixerBlock`.  What exactly this mask will look like will depend on whether we expand the inter-token convolution transformations or not, by which is meant mirorring the two-layer linear transformation (with nonlinear activation) that define the `FeedForward` modules. The similarities may be shown as follows:

```python
def FeedForward(dim, expansion_factor=4):
	inner_dim = int(dim * expansion_factor)
	return nn.Sequential(
		nn.Linear(dim, inner_dim),
		nn.GELU(),
		nn.Linear(inner_dim, dim)
	)

def ConvForward(dim, expansion_factor=2):
	inner_dim = int(dim * expansion_factor)
	return nn.Sequential(
		nn.Conv1d(dim, inner_dim, 1),
		nn.GELU(),
		nn.Conv1d(inner_dim, dim, 1)
		)
```

where the `ConvForward` module is the transformation between tokens.  

We will train the architecture using the masking approach that is commonly applied to causal language models in which the objective of training is to predict the next token in a sequence.  This means that we need to prevent the model from using information from tokens to the right some token when learning to predict that token.

```python
class LanguageMixer(nn.Module):

	def __init__(self, n_vocab, dim, depth, tie_weights=False):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, dim)
		self.mixerblocks = nn.ModuleList(
			[MixerBlock(
				dim = dim,
				length = tokenized_length,
				)
			for i in range(depth)]
			).to(device)
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
		if tie_weights:
			self.lm_head.weight = self.wte.weight
		self.cel = nn.CrossEntropyLoss()
```


For a recursive neural network model, there is pretty much one way to train the model: for each word in a sequence, get the model's prediction, find the loss, and backpropegate. For transformers and other models we could do the same thing but there is a much more efficient way to train: instead of iterating through all words in an input we instead feed the entire input into the model as if we were going to predict the next word, but instead we take the `lm_head` output for each input, find the loss, and backpropegate the total loss.

There are two important modifications necessary for this more parallelizable training. The first is that we need to have the model compare the output of the model for sequence elements $a_0, a_1, ..., a_{n-1}$ to the element $a_n$ during the loss calculation, meaning that we need to shift the target element $a_n$ such that it is accessed when the model is at position $a_{n-1}$. This is accomplished in the forward pass of the model (below) by shifting the labels (the target elements) to start with the input element at index 1 rather than index 0 in `labels[..., 1:].contiguous()`. The model's outputs are clipped such that the last output (which would correspond to the next element after the end of the input and does not exist in the input itself) is omitted. For compatibility with the HuggingFace `trainer`, we compute the cross-entropy loss in the forward pass and supply the value as part of a tuple with the output itself.

The line `labels = rearrange(labels, 'b p t -> b (p t)')` in the forward pass may be unfamiliar with those who have not worked with vision models in the past. For whatever reason, Einstein sum tensor manipulation never became as popular in the language modeling world as for vision models. There are certainly pros (succinct notation) and cons (portability) to using `einsum` notation, but we will use `einsum` mostly for reshaping tensors. For example, `labels = rearrange(labels, 'b p t -> b (p t)')` simply removes an index dimension of our tensor between the `batch` and `token` dimensions and could also be accomplished with `labels = torch.squeeze(labels, dim=1)` but is arguably more expressive.

```python
class LanguageMixer(nn.Module):
	...
	def forward(self, input_ids, labels=None):
		x = input_ids
		x = x.to(device)
		x = self.wte(x)
		for block in self.mixerblocks:
			x = block(x)
		output = self.lm_head(x)
		labels = rearrange(labels, 'b p t -> b (p t)')
		output = rearrange(output, 'b t e -> b e t')
		shift_logits = output[..., :-1].contiguous()
		shift_labels = labels[..., 1:].contiguous()
		loss = self.cel(shift_logits, shift_labels)
		return loss, output
```

Besides shifting the output, we need a second addendum for causal language modeling: only information from previous tokens $a_0, a_1, ..., a_{n-1}$ must reach token $a_n$ and not information from succeeding tokens $a_{n+1}, a_{n+2}, ...$. For the transformer architecture it is customary to mask the softmax values of the $KQ^T$ to only use information from query projections of past tokens, but as we are using a 1-dimensional convolution transformation with no softmax a different approach will be necessary.

Instead we shall mask the weights of the convolution such that the only non-zero weights supplied to $a_n$ will originate from $a_0, a_1, ..., a_{n-1}$. How this may be done is easier to see if we look at only one convolution between token: given an input matrix $X \in \Bbb R^{m, n}$ with $m=3$ tokens and $n=2$ features per token, 

$$
X = 
\begin{bmatrix}
x_{0, 0} & x_{0, 1}\\
x_{1, 0} & x_{1, 1}\\
x_{2, 0} & x_{2, 1}\\
\end{bmatrix}
$$

if we are given convolution weights from a single filter layer

$$
W_0 = 
\begin{bmatrix}
2\\
1\\
0\\
\end{bmatrix}
$$

we get the output (ie one token)

$$
X \circ W_0 = \\
\\
\begin{bmatrix}
2x_{0, 0}+1x_{1, 0}+0x_{2, 0} & 2x_{0, 1}+1x_{1, 1}+0x_{2, 1}\\
\end{bmatrix}
$$

Likewise, with two convlutional feature weight layers we perform the same operation with the second to recieve a 2x2 output and for three we have a 3x2 output. If we concatenate the weight layers together in a single matrix such that each column weight becomes a matrix column, we want to use an upper triangular mask: in this case, the convolutional weight matrix $W$

$$
W = 
\begin{bmatrix}
2 & 1 & 1\\
1 & 1 & 4\\
1 & 3 & 1\\
\end{bmatrix}
$$

becomes the masked weight matrix

$$
m(W) = U \circ W
\\
m(W) = 
\begin{bmatrix}
2 & 1 & 1\\
1 & 1 & 4\\
1 & 3 & 1\\
\end{bmatrix}

\circ

\begin{bmatrix}
1 & 1 & 1\\
0 & 1 & 1\\
0 & 0 & 1\\
\end{bmatrix}

= 

\begin{bmatrix}
2 & 1 & 1\\
0 & 1 & 4\\
0 & 0 & 1\\
\end{bmatrix}
$$

such that now for the first token we have the output

$$
X * m(W) =  \\

\begin{bmatrix}
x_{0, 0} & x_{0, 1}\\
x_{1, 0} & x_{1, 1}\\
x_{2, 0} & x_{2, 1}\\
\end{bmatrix}

*

\begin{bmatrix}
2 & 1 & 1\\
0 & 1 & 4\\
0 & 0 & 1\\
\end{bmatrix} 

\\
\\
= 
\\
\\

\begin{bmatrix}
2x_{0, 0}+0x_{1, 0}+0x_{2, 0} & 2x_{0, 1}+0x_{1, 1}+0x_{2, 1}\\
1x_{0, 0}+1x_{1, 0}+0x_{2, 0} & 1x_{0, 1}+1x_{1, 1}+0x_{2, 1}\\
1x_{0, 0}+4x_{1, 0}+1x_{2, 0} & 1x_{0, 1}+4x_{1, 1}+1x_{2, 1}\\
\end{bmatrix}
$$

which is what we want, as each token recieves non-zero weights from preceeding (after shifting) tokens only.

In our implementation we actually use a lower-triangular mask (`tril`) because we must first re-arrange each convolutional weight tensor into a single weight matrix, and by default our rearrangement places each convolution weight column as a row in our collected matrix, ie it is transposed.

```python
rearranged_shape = rearrange(self.conv.weight, 'f d p -> f (d p)').shape
mask = torch.tril(torch.ones(rearranged_shape)).to(device)
applied_mask = rearrange(self.conv.weight, 'f d p -> f (d p)') * mask
self.conv.weight.data = rearrange(applied_mask, 'f (d p) -> f d p', p=1)
```

Thus we modify each 1D convolution in the mixer such that the convolutional weight is lower-triangular and perform the same operation as before, 

![masked mixer]({{https://blbadger.github.io}}/deep-learning/masked_llm_mixer.png)

but bear in mind that the full mixer has two convolutions separated by a nonlinearity (`self.expand_conv=True` below) so we actually need to perform the reshaping and masking for both convolutions.

```python
class MixerBlock(nn.Module):

	def __init__(self, dim, length, mixer_mask=True, expand_conv=True):
		super().__init__()
		...

	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')

		# for CLM training, apply lower triangular mask to convolution weights
		if self.mixer_mask:
			# Mask logic here
			...
		residual = x
		x = self.seq_layernorm(x)
		x = self.conv(x) + residual
		residual = x
		x = self.patch_layernorm(x)
		x = self.patch_ff(x) + residual
		return x
```

### Training

We make use of the `transformers.trainer()` module, which has a couple very useful features for ease of use: automatic logging checkpointing the model and optimizer states, masking loss on the pad token, etc. For testing purposes I also used the following barebones trainer (note that this does not mask pad token loss and should not be used for an actual training run):

```python
def train_model():
	model.train()
	total_loss = 0
	for step, batch in enumerate(train_data):
		batch = rearrange(batch, 'b (p t) -> b p t', p=1)
		optimizer.zero_grad()
		batch = batch.to(device) # discard class labels
		loss, output = model(batch, batch)
		total_loss += loss.item()
		loss.backward()
		optimizer.step()
	print ('Average loss: ', total_loss / len(batch))
```

To make results comparable, we use the same tokenizer, dataset (TinyStories), and batch size (16) and observe the training and evaluation cross entropy loss for the Llama-style transformer compared to our MLP mixer.  We perform training runs of around 12 hours on an RTX 3060.  

It should be noted that the transformer requires substantially more memory to store the gradients, optimizer, and parameters than the mixer: given a batch size of 16, a llama model with $d_{model}=128$ and $n=8$ exceeds 10 GB vRAM during training compared to the 2.4 GB vRAM required for a mixer of the same $n$ and double the $d_{model}$, both for a context window size of 512.  This is mostly due to the $O(n^2)$ complexity of the transformer model as the context window increases compared to the $O(n)$ complexity inherent in the language mixer. It also stems from the more 'efficient' use of gradients by the mixer, as gradient do not need to pass along non-trainable parameters as is the case for transformers (where attention gradients travel from $KQV$ projections to the $KQV$ values themselves and back). Thus we cannot compare these models directly using only $d_{model}$ and $n$, but instead use a ballpark figure for these and compare training and test vRAM.

It should also be noted that optimizing a mixer of a similar size to a transformer requires much less time: one typically sees between 10x and 20x more time required for a transformer with $d_{model}=128$ and $n=8$ compared to a mixer with twice the $d_{model}$ and the same number of blocks.  This means that the Chinchilla scaling laws applicable to transformer architectures are expected to be much more favorable for MLP mixers, at least for their

Now we train the models, using an approximately fixed compute budget (12 GB vRAM and 12 hours on an RTX 3060).  We find that the masked mixer with the parameters detailed above achieves a substantially smaller training (2.169) and validation (2.201) loss after this twelve hours than the Llama model (2.497 validation and 2.471 training loss).  This is mostly because the mixer is around six times as fast to train as the transformer: there is nearly identical loss if we compare equal steps (2.497 versus 2.505 validation loss at step 160000). Equivalent steps mean little for language models, however, as they are inherently resistant to overfitting (see that there is minimal overfitting after 5.6 epochs at twelve hours of training for the mixer) such that simply increasing the dataset size or number of epochs in this case yeilds lower training and validation loss than training a larger model with an effectively smaller dataset.


### Mixer Inference

Low training and validation loss are useful guides, but the goal of this research is to find architectures that are more efficient than the transformer at causal language modeling, which is simply to generate the next word in a sentence.  

It is tempting to simply take the last output of the mixer model as the next token and perform a sliding window on the input to keep a constant number of tokens (which the mixer requires, unlike a transformer model). But there is a problem with this method: the last output is not actually trained at all! Recall that the model used a shifted logits approach, such that the trained output is $[..., :-1]$ which does not include the last element because that is the prediction for the next token after the input ends, information that is absent.

Instead we can have the mixer simply replace token after token, maintaining the causal language mask after the last supplied token as follows:

![masked mixer generation]({{https://blbadger.github.io}}/deep-learning/masked_mixer_gen.png)

For 50 generated tokens at the end of the 512-length context window, this corresponds to the following inference code:

```python
fout = []
for i in range(50, 1, -1):
	loss, output = model(tokens, labels=tokens)
	out_token = torch.topk(output, dim=1, k=1).indices.flatten()[-i]
	tokens[..., -i+1] = out_token
```

while maintaining the `mixer_mask=True` flag on the model.

The outputs are extremely good: for our mixer of size $d_{model}=256$ and $n=8$ blocks (1.8 GB vRAM for 16-bit training on 512 context tokens), trained for 12 hours on TinyStories, we have for the validation dataset input (`<unk>` tokens are newlines)

"""
Once upon a time, there was a little boy named Tim. Tim had a big, orange ball. He loved his ball very much. One day, Tim met a girl named Sue. Sue had a pretty doll. Tim liked Sue's doll, and Sue liked Tim's orange ball.<unk>Tim and Sue thought about a trade. They would trade the ball for the doll. Tim was not sure. He loved his orange ball. Sue said, "I promise to take care of your ball. You can play with it when you
"""

The actual next 50 tokens to this input (generated by ChatGPT) are

"""
want." Tim said, "I promise to take care of your doll too."<unk>They did the trade. Tim played with the doll, and Sue played with the orange ball. But then, Sue's big dog came. The dog saw the orange ball and thought it was a toy. The dog bit the ball and it popped
"""

and our langauge mixer supplies the following output:

**`
come back."<unk>Tim thought about it and agreed. Sue gave the ball back to Tim. They played together, and Tim was very happy. Sue was happy too. They learned that sharing can make everyone happy. And that is how the orange ball can bring happiness to everyone.
`**

Compare this to the greedy generation output from the Llama transformer model $d_{model}=128$ and $n=8$ blocks (10.7 GB vRAM for 16-bit training on 512 context tokens), trained for 12 hours on TinyStories: 

**`
'll find your ball."<unk>Tim was happy. He said, "Thank you, Sue. You are a good friend. You are a good friend." Sue gave Tim a new ball. Tim gave Sue a new ball. Sush was blue and shiny. Sush, Spot, and Spot. They played with their ball
`**

although this turns out to be a bad prompt for the transformer. A better one is as follows: 

"""
One day, a little boy named Tim went to play with his friend, Sam. They wanted to play a game with a ball. The game was to see who could get the best score.<unk>Tim threw the ball and Sam tried to catch it. Sam missed the ball and it went far away. Sam was angry. He said, "I don't like this game. I want to lie down and rest."<unk>Tim said, "Okay, let's lie down and rest. Then we can try again to get a good score." They lay down under a big tree and looked at the sky. They saw birds flying and clouds moving.<unk>After a while, they got up and
"""

The masked mixer completes this story as follows:

**`
played a game of tag. Tim was fast, but Sam was faster. They ran and ran until they reached the finish line. Tim was tired but happy. Sam said, "Good job, Sam!" Sam smiled and said, "Good job, Tim!" They both smiled and hugged each other.
`**

The transformer completes this as

**`
went home. They sat down on the grass and closed their eyes. They closed their eyes and fell asleep. Tim dreamed of the snow he found his way home. He felt safe and happy in his new score. He knew that he could always count on his score again. And he lived happily ever after.
`**

which is a little less coherent. Thus we see that there is indeed an improvement in output quality, reflective of the lower training and evaluation loss of the trained mixer compared to the transformer.

### Flat Mixers Train Faster

The results in the last section were obtained without any hyperparameter optimization or architectural tuning methods, so it stands to reason that simply tweaking the model architecturs could result in smaller fixed-compute (12 hours on an rtx 3060 given 12 GB vRAM) loss. If the llama model width is increased to $d_{model}=256$, the loss after 12 hours is lower than for the $d_{model}=128$ version (1.99 and 2.03 training and validation, respectively), which

When examining the ability of a model to [represent](https://blbadger.github.io/language-discreteness.html) its input, it was found that input representation of tokens whose output was masked (called 'non-self' tokens on the linked page) is accurate for untrained masked mixers if either the model were relatively large ($d_{model}=1024, \; n=24$ or $d_{model}=2048, \; n=8$) or else if the model remained small $d_{model}=512, \; n=8$ but the expanded sequential convolutions were replaced with a single convolution as shown in a previous section on this page.  

This suggests that these 'flat' mixers may be able to be trained more effectively than the expanded convolution mixers or transformers, and it turns out that this appears to be the case: a $d_{model}=512, \; n=8$ flat mixer (requiring 3.59 GB vRAM to train on a context window of 512 tokens) achieves a training loss of 1.842 and a validation loss of 1.895 after 12 hours on TinyStories, which is lower than the $d_{model}=256, \; n=8$ transformer that requries more than double the memory (1.99 and 2.03 training and validation, respectively). Even one compares loss at a fixed number of updates, the flat mixer still outperforms the $d_{model}=128$ transformer by a wide margin: 2.116 versus 2.471 loss, respectively, at 114000 steps. This is most notable as the flat mixer has around a third of the 'effective' parameters that the transformer has. 

The largest 8-layer transformer that will fit in 12GB vRAM (at 512 token context with a 16-size minibatch) has a $d_{model}=256$, and at 12 hours on a 3060 this achieves 1.99 and 2.03 training and validation loss, respectively. This lags behind the training and validation loss of 1.81 and 1.86 of a $d_{model}=1024, \; n=8$ mixer trained with the same total compute budget. 

For the same prompt in the last section ("One day, a little boy..."), the trained flat mixer yields

**`
played a game of catch. Tim threw the ball to Sam, and Sam caught it. They laughed and played until the sun went down.<unk>At the end of the day, Tim and Sam were tired but happy. They went home and took a nap. They dreamed of playing catch again tomorrow. And they did.
`**

which is a more coherent and plot-accurate completion than either the transformer or even the expanded mixer, again reflective of a (much) lower training and validation loss than either architecture.

### Scaling Properties

The masked mixer architecture gives us an easy way to modify the number of inter-token weights (1D convolution `ConvForward` weights) as well as the number of intra-token weights (the `FeedForward` weights). We can observe the loss achieved by varying the number of each type of parameter independently, a feat which is much more difficult to pull off for the transformer as the number of $K, Q, V$ projection weights are usually tied to the $d_{model}$. 

Which type of weight is likely to be more important: that is, given a fixed number of total weights should we allocate more to intra- or inter-token parameters for the lowest loss given a fixed amount of compute?  When considering the causal langauge generation process, there are arguments to be made for both types, as clearly complex relationships between words are just as important if not moreso than a nuanced understanding of a word itself.

One argument for the importance of allocating more parameters to intra-token weights is that all information from all previous words must pass through these weights (ignoring residual connections for the moment), whereas inter-token weights may add information from many parts of an input over many layers.

### Implications

Seeking to make a more efficient learning algorithm than a transformer, we used the observation that token representation is superior for modified MLP-Mixer architectures to craft a model capable of replicating the autoregressive language generation of GPT-style decoder-only transformers.

It is worth restating the more noteworthy findings of this work as concisely as possible:

1. An unoptimized flat masked mixer achieves much lower loss than an effectively larger transformer at a given update step.
2. Given equal compute, this same mixer reaches a much lower training and validation accuracy which is reflected in its autoregressive output relative to the transformer's output.
3. Our mixer implementation uses no traditional regularization techniques (but does not overfit to any greater degree than the transformer), instead relying on the intrinsic generalization inherent in gradient descent-based optimization of high-dimensional space (see [this paper](https://arxiv.org/pdf/2211.09639.pdf) for more on this subject) combined with the 'inherent' regularization in language datasets.
4. This is all possible without innovations that are now used nearly ubiquitous for transformers such as attention, rotary positional encoding (or any explicit positional encoding at all) or weight tying between the embedding and langauge modeling head
5. A masked mixer trained using approximately 1/18th the compute of a transformer model published in the TinyStories paper achieves a similar human evaluation score on that dataset







