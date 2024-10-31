## Masked Mixers II

We continue exploring topics from [Part I](https://blbadger.github.io/smaller-lms.html), with a more thorough version of these results found in [this paper](https://arxiv.org/pdf/2409.01482)
Application of masked mixers to larger datasets will be explored, theory is expanded, and the nature of the learned structure of masked mixers versus transformers is investigated.

### Why Masked Mixers?

In [Part I](https://blbadger.github.io/smaller-lms.html) the motivation behind masked mixers for language modeling was detailed, but can be restated briefly: from high-dimensional projection theory (specifically the Johnson-Lindenstrauss lemma) we can accurately represent points in high-dimensional space in a space with much fewer dimensions (approximately the log of the number of points we have). This is an important result here because it implies that one can make language models that are capable of fitting extremely large datasets with what is currently considered an extremly small number of parameters: for example, a dataset one million times the size of the already large 15 trillion token Llama-3.1 dataset ($1.5 * 10^19$ tokens to be precise) may be arbitrarily well fitted by a 352-dimensional model, which would require around 100 million parameters for a llama-style generative model.

Empirically it appears that the current state-of-the-art model type (the transformer) trains much too slowly to achieve this feat in any reasonable length of time, and the best-performing models are typically trained on thousands of GPUs for many days. From other investigations on information transfer between model layers, we wondered whether an architecture that more accurately represents its inputs (the masked mixer) would learn more efficiently.

Restricting our investigations to a small dataset (TinyStories, ie short children's stories written by ChatGPT) we found that although masked mixers are more efficient learners than the original GPT-style transformers for the task of causal language modeling, highly optimized and current transformers learn even more efficiently.  On the other hand, masked mixers were found to be much more efficient for language retrieval which is expected from their superior input representation properties.

### Accuracy and Flexibility

Even with these promising features, the question may be asked: why masked mixers? Before proceeding to further investigations on this architecture, it is worth considering what you get when you swap self-attention for masked convolutions.

Before masked mixers were first trained, it was found that these models give much more accurate representations of their inputs than transformers. In effect, this means that given an input, the information necessary to uniquely identify that input via gradient descent is retained throughout the mixer but not the transformer. It could be argued for or against the idea that this would be useful for language generation, and perhaps more convincingly argued that this is important for langauge retrieval. But accurate input representation remains a useful feature of mixers for a variety of applications. We will refer to this input representation *accuracy* more in the future.

Perhaps as important is the model architecture *flexibility* of masked mixers. From investigations into representation in [vision transformers](https://blbadger.github.io/vision-transformers.html), it was apparent that vision transformers require each key component of their architectures: layer normalizations, MLPs, self-attention, and residual connections are all required for effective gradient propegation.

This can be tested more directly by simply removing each component and observing the training process. Ironically for an architecture introduced as 'Attention is all you need', self-attention is actually the only removable component (as long as it is replaced by some other trainiable inter-token parameters): removal of MLPs (or replacing with attention) layer norms or residual connections results in very poor language model training: either failure to minimize a loss function (if MLPs are removed or replaced with attention) or training becomes unstable (for removal of layer norms or residuals) and loss spikes to infinity. The reason for this is that attention is a rather difficult transformation for gradients to propegate across, and this is important because it essentially 'fixes' the architectures of models with attention to similar patterns, all requiring some form or other of the same components transformers have (layer norms, residuals, MLPs, ets.).

On the other hand it turns out that langauge training proceeds perfectly well (although slightly less efficiently) when layer norms or residuals are removed from the masked mixer architecture. This means that the mixer architecture is effectively much more flexible than the transformer, and can be modified to a much greater extent. This topic will be explored more in the 'Linear mixer' section of this page.

### Bidirectional Language Modeling

We have seen that attention appears to confer benefits to next token prediction, and theoretically this can be expected due to the many-to-one map inherent in this task as well as the inherent noise in real language (not every word necessarily follows from the last, but may be chosen at will). It may be wondered which of these two has a greater influence on the abilities of language models, a question that in this case may be rephrased as follows: is it the inherent noise present in language that is most responsible for the greater efficiency of transformers versus mixers in CLM tasks, or else is it simply the type of mapping performed which is many-to-one?

A simple way to start to address this question is to perform bidirectional language modeling. The justification for this is that with tokens present both before and after the token we want the model to infer, there is in some sense less inherent noise in the language in that there is a greater likelihood that a logical rule may be made to pick one and only one token than there would be if only left-side tokens are present. On the other hand, bidirectional modeling results in an even greater number of tokens that are mapped to a single token than CLM training (twice as many on average) and thus somewhat exaggerates the many-to-one mapping phenomenon. The idea is that if many-to-one mapping is the largest source of the transformer-mixer difference then the mixer will be comparatively worse at bidirectional language modeling than it was for causal language modeling, either reaching or exceeding transformer efficiencies. Conversely, if language noise is the major source of efficiency difference then mixer training efficiency would be comparatively better than for CLM tasks.

In causal language modeling sequence of tokens is used to predict one 'next' token, ie the token to the right for English text. There are other methods of langauge generation, however. In particular one can model language without any causal language masking, instead masking certain tokens themselves such that the task is essentially to infer these few tokens which is analagous to the grade school task of 'fill in the blank'. This token masking approach typically proceeds by masking up to 15% of tokens in a corpus, and using the masked tokens' identities as the labels, and this is the method for BERT, RoBERTa, DeBERTa and other such model training.

It is straightforward to see that this is not a particularly efficient method of training, however: because only 15% of tokens are masked and inferred by the model, only 15% of tokens are trained on during each forward pass. Contrast this with the all-next-token training method for causal language modeling in which all tokens are trained upon each forward pass, and we can estimate that all-next-token training has approximately $1 / (5/33) = 33/5 = 6.6$ times the throughput of this traditional masked langauge modeling approach. This is not an easily ameliorated problem for masked langauge modeling, however, as if too many tokens are masked then the learned distribution becomes too far from the goal of inferring usually one or a few tokens at most.

To train more efficiently, we can instead mask no tokens while maintaining the all-next-token approach of causal language modeling, but apply this method in both the forward and reverse directions simultaneously with some careful tensor shifting. This means that every token is trained upon each forward pass, as each 'next' token is both 'next' in forward and reverse directions. There are a number of different methods to implement this idea, and as we will focus on two that require some care to do so. Firstly, we will examine a masked mixer implementation before proceeding to a Llama-style transformer.

Recall that the causal language modeling masked mixer uses a lower-triangular mask on the inter-token convolutional weights to prevent information from right-indexed tokens from moving 'backward' and influencing next token prediction. We can use the exact same implementation for the 'forward' direction but as we now want information from tokens to the right of our predicted token (but importantly not that token itself) to be used, we can include convolutions with weights connecting tokens $t_{n+2}, t_{n+3}, ..., t_{N}$ to $t_n$. Note again that $t_{n+1}$ remains masked, as this is the token we are attempting to predict. This can be depicted as follows:

![bidirectional mixer](/deep-learning/bidirectional_mixer.png)

A naive implementation of which (using two separate convolutional weight matrices rather than one for clarity) is

```python
class DoubleMixerBlock(nn.Module):

	def __init__(self, dim, length, clm_mask=False, expand_conv=False):
		super().__init__()
		self.patch_layernorm = nn.LayerNorm(dim)
		self.seq_layernormf = nn.LayerNorm(dim)
		self.seq_layernormr = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim)
		self.convf = nn.Conv1d(length, length, 1)
		self.convr = nn.Conv1d(length, length, 1)

	def forward(self, x: torch.Tensor):
		masked_convf = torch.tril(rearrange(self.convf.weight, 'f d p -> p f d'), diagonal=0)
		self.convf.weight.data = rearrange(masked_convf, 'p f d -> f d p').contiguous()

		masked_convr = torch.triu(rearrange(self.convr.weight, 'f d p -> p f d'), diagonal=2)
		self.convr.weight.data = rearrange(masked_convr, 'p f d -> f d p').contiguous()

		residualf, residualr = x, x
		y = x.clone()
		x = self.seq_layernormf(x)
		x, y = self.convf(x) + residualf, self.convr(y) + residualr
		residualf, residualr = x, y
		x, y = self.patch_layernorm(x), self.patch_layernorm(y)
		x, y = self.patch_ff(x) + residualf, self.patch_ff(y) + residualr
		return x + y
```

At first glance it seems that we can just substitute this `DoubleMixerBlock` for a normal mixer block and proceed, but doing so for all models containing more than one mixer block leads to very fast loss minimization towards the origin (CEL $L(O(a, \theta), y)<0.1$ in less than half an epoch) suggesting that something went wrong. Closer examination of the figure above shows what happens: in that case, information from token $t_2$ reaches $t_0$ during the reverse convolution step, which is what we want. But then consider that in the next layer, information travels from hidden layers of $t_0 \to t_1$ and as the model predicts $t_2$ the hidden layer $t_1$, it learns a trivial mapping of that output. This phenomenon is perhaps easier to appreciated diagramatically, and can be shown as follows:

![bidirectional mixer](/deep-learning/bidirectional_mixer_explained.png)

This problem is general to any token pair, and is not even specific to mixers as a bidirectional transformer experiences the same problem. Happily there is a simple solution: observe that two sequential convolutions are required for $t_{n+1}$ information to pass to $t_n$, which is why a model with only one mixer block will not rapidly minimize its loss function as a two or more blocked mixer will. Without loss of generality, one reverse and one forward convolution is required sequentially for this transfer.  Therefore this problem may be avoided by placing all forward and reverse convolutions in parallel rather than in sequence, which can be implemented by having each block take both $x, y$ and return a tuple of both,

```python
class DoubleMixerBlock(nn.Module):
	...
	def forward(self, x: torch.Tensor, y: torch.Tensor):
		...
		return x, y
```

where the linear combination only occurs after all blocks have been passed. The forward pass for the double-sided mixer becomes

```python
class LanguageMixer(nn.Module):
	...
	def forward(self, input_ids, labels=None, **kwargs):
		x = self.wte(x)
		y = torch.clone(x)
		for block in self.mixerblocks:
			x, y = block(x, y)
		output = self.lm_head(x + y) # combine after mixer blocks
		# shift and CEL computation
```

An analagous implementation may be made for a transformer, but this would require rewriting the causal langauge model mask on scaled dot-product attention layers to act in reverse with a changed $torch.triu$ diagonal. A simpler method is to keep the transformer blocks as they are normally implemented and instead reverse the sequence of $y$ such that the 'reverse' block sees tokens in reverse order via `torch.flip()`, maintaining a left-to-right causal mask. One can then undo the reversed `y` order in the token dimension and shift the forward and reverse last hidden layers such that their sequence indices align, add the result, and perform language model head linear transformation and loss calculation. Note that one does not need to truncate the labels as normally occurs, as the $t_0$ is predicted only in the reverse direction. A diagram showing how this works for a sequence of four tokens is given below.

![bidirectional mixer](/deep-learning/bidirectional_transformer_explained.png)

An implementation of this is as follows:

```python
class BidirectionalTransformer(nn.Module):

	def __init__(self, n_vocab, dim, forward_model, reverse_model):
		super().__init__()
		...
		self.forward_model = forward_model # transformer blocks only, no wte or lm_head
		self.reverse_model = reverse_model # transformer blocks only, no wte or lm_head

	def forward(self, input_ids, labels=None, attention_mask=None):
		x = input_ids
		x = x.to(device).squeeze(1)
		x = self.wte(x)
		y = torch.flip(x.clone(), dims=[1]) # reversed in token dim
		
		forward = self.forward_model(x)
		reverse = self.reverse_model(y)
		pad = torch.zeros(x.shape[0], 1, x.shape[2]).to(device)

		reverse = torch.cat([torch.flip(reverse, dims=[1])[..., 1:, :], pad], dim=1) # right pad reverse
		forward = torch.cat([pad, forward[..., :-1, :]], dim=1) # left pad forward

		output = self.lm_head(forward + reverse)
		logits = rearrange(output, 'b t e -> b e t')
		loss = self.cel(logits, labels)
		return loss, output
```

When we compare mixer versus transformer performance on bidirectional token prediction versus causal language modeling-style next token prediction, we see that the relative performance is nearly identical. 

![uni vs bidirectional](/deep-learning/uni_vs_bidirectional.png)

### Masked Mixers make better Autoencoders than Transformers

The accurate input representation present in masked mixers suggests that these models retain more information from their inputs than is present in transformers. It appears that next token prediction does not require or indeed is not particularly benefitted by this increased information compared to the focus brought by attention, but it was hypothesized and subsequently observed that masked mixers are far superior retrieval models as this task would be expected to require more information. 

There is a perhaps more direct way to test the hypothesis that masked mixers contain more input information than transformers: we can modify the causal language modeling architectures of the masked mixer and transformer for the task of autoencoding an input. In particular, we want these models to learn a non-trivial autoencoding and not simply return each input token in the output. To do this we can use an encoder-decoder architecture but pass only the last hidden layer of the last token of the encoder to the decoder. For the masked mixer, this may be portrayed as follows:

![autoencoder architecture](/deep-learning/mixer_autoencoder.png)

This is perhaps the most direct way to maintain the parallelization afforded by all-next-token training for a non-trivial autoencoder. For a masked mixer-based

```python
class AutoencodingMixer(nn.Module):
  ...
	def forward(self, input_ids, labels=None):
		... # word-token eembedding
    ... # encoder blocks

		encoder_embedding = x[:, -1, :].unsqueeze(1) # dim=[batch, token, hidden]
		x = encoder_embedding.repeat(1, self.tokenized_length, 1)

		... # decoder blocks
    output = self.lm_head(x)
		labels = rearrange(labels, 'b p t -> b (p t)')
		output = rearrange(output, 'b t e -> b e t')
		loss = self.cel(output, labels)
		return loss, output
```

For a llama-style transformer, this architecture can be implemented as follows: first we modify the LlamaModelForCaualLM to take embeddings rather than tokens and supply the necessary positions

```python
class AbbreviatedModel(nn.Module):

	def __init__(self, model, depth=8, tokenized_length=512):
		super().__init__()
		self.model = model
		self.depth = depth
		self.position_ids = torch.tensor([[i for i in range(tokenized_length)]]).to(device)

	def forward(self, input_ids: torch.Tensor):
		x = input_ids
		position_ids = self.position_ids.repeat(input_ids.shape[0], 1)

		for i in range(self.depth):
			x = self.model.model.layers[i](x, position_ids=position_ids)[0]
		return x

# initialization
encoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=tokenized_length)
decoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=tokenized_length)
```

and then the autoencoder is implemented in the same manner as the mixer autoencoder.

Recall that masked mixers contain far fewer inter-token parameters and thus may be trained with a much larger $d_m$ size while maintaining other architectural constraints identically to transformers for fixed memory, and mixers of identical architectural 'sizes' train much more quickly. With this in mind, we can first observe autoencoding performance for identically-sized models: given a $d_m$=512 and $n_l=8$ (ie 8 encoder layers and 8 decoder layers). After 2.25 of TinyStories training, the masked mixer autoencoder reaches train/test losses of 4.53/4.35 respectively whereas the same-dimensional transformer only manages losses of 5.31/5.23. For $d_m=1024, n_l=4$ (the largest $d_m=1024$ transformer that fits in V100 memory) reaches 5.05/4.99 train/test loss after three epochs, whereas a masked mixer autoencoder of the same $d_m, n_l$ reaches 3.85, 3.66 (below).

These are very large performance gaps: recall that the difference between transformer and mixer CLM loss is typically 0.5-2%, such that with a modest increase in training duration one architecture is able to achieve the loss of the other. But from the figure below it is apparent that it would take a huge number of steps (perhaps 1000x) for the transformer to match the mixer's loss achieved, if it ever is.

![autoencoders](/deep-learning/language_autoencoders.png)

The gap is even larger when we consider that the mixer occupies a much smaller memory footpring for identical $d_m, n_l$ parameters. If we match the mixer to the $d_m=1024, n_l=4$ transformer's memory on device by doubling the $n_l \to 8$, the mixer reaches 1.65/1.37 train/test loss using the same compute (4x V100s, 6h) as the above transformer. This would be expected to require hundreds (!) of epochs for the transformer to match, and in that way one could claim that the mixer is hundreds of times as efficient an autoencoder as a transformer.

### Language Generation Training Efficiency

The goal of a machine learning algorithm is to minimize some loss function on a dataset efficiently, and the hope is that the minimization process and dataset are sufficient to generalize to the task you actually want to perform (typically representation by a 'test' or 'evaluation' dataset). The choice of a loss function, the model to use, the optimization approach, and the dataset are all important factors in whether the generalization actually occurs.

### Linear Mixers

In the field of numerical analysis one can generally say that there are a number of differences between linear and nonlinear processes, at least at a very high level. Perhaps most notably, linear transformations may be completed in one effective 'step' whereas nonlinear transformations require many 'steps'. Whether this is accurate or not in practice is somewhat dubious, but for our particular application it will indeed be.

This is relevant because we can get an idea of how to make an extremely fast (to run, that is) language model by considering what exactly happens during autoregressive inference. When one considers autoregressive inference, it is generally noted that models like Transformers that compare all tokens to all other tokens scale with $n^3d$ time complexity without caching, and $n^2d$ with caching for $n$ tokens and hidden dimension $d$. It is less appreciated that inference time also depends on the number of layers of linear transformations in a model $l$, as because typically each layer is separated from each next layer by one or more nonlinear transformation (layer normalization, ReLU, GeLU etc.) such that the actual time complexity becomes $n^2dl^2$ as each of the $n^2$ token comparisons require $l$ steps. 

Clearly it would be advantageous to reduce the number of layers in a model, but how can this be done while maintaining an efficiently trainable architecture? 





