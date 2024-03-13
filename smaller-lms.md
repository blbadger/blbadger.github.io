## Smaller, Simpler Langauge Models

### Background 

The training of the most effective language models today (3/2024) requires an enormous amount of computational resources: a whopping 1720320 hours of 80GB nvidia A100 compute time were required to train the 70 billion parameter version of [Llama 2](https://arxiv.org/pdf/2307.09288.pdf). Assuming that the meta RSC was used (6080 GPUs), this comes out to nearly two weeks of nonstop training for that entire cluster.  The cost to reproduce this training on the cloud at the same day would be around four million usd., making this kind of training prohibitively expensive for all except a few large organizations.

This prohibitive amount of compute power required is mostly down to the very large size of the models that are currently trained: most LLMs are trained on something like 2 trillion tokens of text, but this is not actually that much information when one considers that each token is typically in bytecode and therefore the training dataset is around 2 TB, substantially smaller than the large image datasets required to train (much smaller) diffusion models.

To understand why this is the current state of affairs, it is enlightening to remember firstly that all current large language models are based on the transformer architecture, and that that architecture was origially introduced with the motivation of looking for a model that would not 'saturate', is it gets better the more tokens one uses to train it and the larger the model is. It is therefore not very surprising that the current large language models are very large indeed, as the architecture they are built on was originally found to benefit from an enormous number of parameters.

This begs the question: are these parameters necessary? It is clear that transformer-based models do indeed become substantially more effective with an increase in the number of parameters they contain, but if we were not restricted to one particular architecture is it possible that we could design a model with far fewer parameters for language modeling?

A very quick calculation suggests that billions or even millions of parameters are far more than would be necessary to model the English language. It has been claimed that there are somewhere around $10^{570}$ possible English sentences, as an upper bound. Without knowing how to model these sentences, we can view them as unique points in an arbitrarily high-dimension space. Now due to a theorem of high-dimensional space, the same points may be obtained with arbitrary precision in a space that is approximately $\log 10^{570} = 570$ dimensional.  This means that a model with the same number of parameters may exist such that each sentence may be found via a combination of these parameters. What type of model could this be? 

### Introduction

The purpose of this page is to detail a number of experiments on the use of alternate architectures to the currently 

The experimental setup will be as follows: for our dataset we will start with TinyStories, which is a collection of simple text that contains a limited vocabulary similar to what a four-year-old would use that allows for effective modeling by transformers in the millions rather than billions of parameter scale. For reference, small versions of state-of-the-art transformer models will be trained and used as a comparison to the new architectures that we will try here.

### Language Mixer from Scratch

[Elsewhere](https://blbadger.github.io/language-discreteness.html) it was observed that transformers exhibit a somewhat unexpected phenomena: firstly that transformer blocks must be extremely wide (embedding size $e > 3000$) in order to have any accurate input representation ability, and secondly that the ability of a transformer to accurately represent a token it has seen previously (ie a 'non-self' token) disappears after training.  On the other hand, a modification of the MLP mixer architecture was found to have accurate self- and nonself- token representation even from very small models with $e < 100$. Thus this may be a good candidate with which to start the process of looking for more effective architectures than the transformer.


The MLP mixer architecture is conceptually similar to a transformer if all the multi-head attention layers were replaced with linear transformations over the sequence, rather than token, dimension.  We modify this architecture for causal language modeling use as follows:

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

![masked mixer]({{https://blbadger.github.io}}deep-learning/llm_mixer.png)

```python
class MixerBlock(nn.Module):
	...
	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')
		if self.mixer_mask:
			if self.expand_conv:
				masked_conv0 = nn.Parameter(rearrange(torch.tril(rearrange(self.conv[0].weight, 'f d p -> f (d p)')), 'f (d p) -> f d p', p=1))
				masked_conv2 = nn.Parameter(rearrange(torch.tril(rearrange(self.conv[2].weight, 'f d p -> f (d p)')), 'f (d p) -> f d p', p=1))
				self.conv[0].weight = masked_conv0
				self.conv[2].weight = masked_conv2
			else:
				self.conv.weight = torch.nn.Parameter(rearrange(self.conv.weight, 'f d p -> f (d p)'))
				self.conv.weight = torch.nn.Parameter(torch.tril(self.conv.weight))
				self.conv.weight = torch.nn.Parameter(rearrange(self.conv.weight, 'f (d p) -> f d p', p=1))
		residual = x
		x = self.seq_layernorm(x)
		x = rotary_emb.rotate_queries_or_keys(x)
		x = self.conv(x) + residual
		residual = x
		x = self.patch_layernorm(x)
		x = self.patch_ff(x) + residual
		return x
```
We will train the architecture using the masking approach that is commonly applied to causal language models in which the objective of training is to predict the next token in a sequence.  This means that we need to prevent the model from using information from tokens to the right some token when learning to predict that token.

![masked mixer]({{https://blbadger.github.io}}/deep-learning/masked_llm_mixer.png)

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











