## Smaller, Simpler Langauge Models

### Background 

The training of the most effective language models today (3/2024) requires an enormous amount of computational resources: a whopping 1720320 hours of 80GB nvidia A100 compute time were required to train the 70 billion parameter version of [Llama 2](https://arxiv.org/pdf/2307.09288.pdf). Assuming that the meta RSC was used (6080 GPUs), this comes out to nearly two weeks of nonstop training for that entire cluster.  The cost to reproduce this training on the cloud at the same day would be around four million usd., making this kind of training prohibitively expensive for all except a few large organizations.

This prohibitive amount of compute power required is mostly down to the very large size of the models that are currently trained: most LLMs are trained on something like 2 trillion tokens of text, but this is not actually that much information when one considers that each token is typically in bytecode and therefore the training dataset is around 2 TB, substantially smaller than the large image datasets required to train (much smaller) diffusion models.

To understand why this is the current state of affairs, it is enlightening to remember firstly that all current large language models are based on the transformer architecture, and that that architecture was origially introduced with the motivation of looking for a model that would not 'saturate', is it gets better the more tokens one uses to train it and the larger the model is. It is therefore not very surprising that the current large language models are very large indeed, as the architecture they are built on was originally found to benefit from an enormous number of parameters.

This begs the question: are these parameters necessary? It is clear that transformer-based models do indeed become substantially more effective with an increase in the number of parameters they contain, but if we were not restricted to one particular architecture is it possible that we could design a model with far fewer parameters for language modeling?

A very quick calculation suggests that billions or even millions of parameters are far more than would be necessary to model the English language. It has been claimed that there are somewhere around $10^570$ possible English sentences, as an upper bound. Without knowing how to model these sentences, we can view them as $10^570$ points in an arbitrarily high-dimension space. Now due to a theorem of high-dimensional space, the same points may be obtained with arbitrary precision in a space that is approximately $\log 10^570 = 570$ dimensional.  This means that a model with the same number of parameters may exist such that each sentence may be found via a combination of these parameters.

What type of model could this be? We will use a combination of representational accuracy and efficiency as our guiding principles and try out some architectures.

### Introduction

[Elsewhere](https://blbadger.github.io/language-discreteness.html) it was observed that transformers exhibit a somewhat unexpected phenomena: firstly that transformer blocks must be extremely wide (embedding size $e > 3000$) in order to have any accurate input representation ability, and secondly that the ability of a transformer to accurately represent a token it has seen previously (ie a 'non-self' token) disappears after training.  On the other hand, a modification of the MLP mixer architecture was found to have accurate self- and nonself- token representation even from very small models with $e < 100$. Thus this may be a good candidate with which to start the process of looking for more effective architectures than the transformer.

The experimental setup will be as follows: for our dataset we will start with TinyStories, which is a collection of simple text that contains a limited vocabulary similar to what a four-year-old would use that allows for effective modeling by transformers in the millions rather than billions of parameter scale. For reference, small versions of state-of-the-art transformer models will be trained and used as a comparison to the new architectures that we will try here.

The MLP mixer architecture is conceptually similar to a transformer if all the multi-head attention layers were replaced with linear transformations over the sequence, rather than token, dimension.  We modify this 

```python
class MixerBlock(nn.Module):

	def __init__(self, dim, length, mixer_mask=True, dropout=0., expand=False):
		super().__init__()
		self.layernorm = nn.LayerNorm(dim)
		self.seq_layernorm = nn.LayerNorm(length)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim, expansion_factor=expansion_factor)
		if expand:
			self.conv = ConvForward(dim, expansion_factor=expansion_factor)
		else:
			self.conv = nn.Conv1d(dim, dim, 1)
		
		# for CLM training: mask conv weights to become upper-triangular
		if mixer_mask:
			if expand:
				self.conv[0].weight = torch.nn.Parameter(torch.triu(self.conv[0].weight))
				self.conv[2].weight = torch.nn.Parameter(torch.triu(self.conv[2].weight))
			else:
				self.conv.weight = torch.nn.Parameter(torch.triu(self.conv.weight))
```

We will train the architecture using the masking approach that is commonly applied to causal language models in which the objective of training is to predict the next token in a sequence. 

```python
class LanguageMixer(nn.Module):

	def __init__(self, n_vocab, dim, depth):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, dim)
		self.mixerblocks = nn.ModuleList(
			[MixerBlock(
				dim = dim,
				length = tokenized_length,
				)
			for i in range(depth)]
			).to(device)
		self.lm_head = nn.Linear(dim, n_vocab)
		self.cel = nn.CrossEntropyLoss()

	def forward(self, input_ids, labels=None):
		x = input_ids
		x = x.to(device)
		x = self.wte(x)
		for block in self.mixerblocks:
			x = block(x)
		output = self.lm_head(x)
```

```python
  def forward(self, input_ids, labels=None):
  ...
		labels = rearrange(labels, 'b p t -> b (p t)')
		output = rearrange(output, 'b t e -> b e t')
		shift_logits = output[..., :-1].contiguous()
		shift_labels = labels[..., 1:].contiguous()
		loss = self.cel(shift_logits, shift_labels)
		return loss, output
```











