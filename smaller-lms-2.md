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

Recall that masked mixers contain far fewer inter-token parameters and thus may be trained with a much larger $d_m$ size while maintaining other architectural constraints identically to transformers for fixed memory, and mixers of identical architectural 'sizes' train much more quickly. With this in mind, we can first observe autoencoding performance for identically-sized models: given a $d_m$=512 and $n_l=8$ (ie 8 encoder layers and 8 decoder layers) 


### Language Generation Training Efficiency

The goal of a machine learning algorithm is to minimize some loss function on a dataset efficiently, and the hope is that the minimization process and dataset are sufficient to generalize to the task you actually want to perform (typically representation by a 'test' or 'evaluation' dataset). The choice of a loss function, the model to use, the optimization approach, and the dataset are all important factors in whether the generalization actually occurs.


### Linear Mixers

In the field of numerical analysis one can generally say that there are a number of differences between linear and nonlinear processes, at least at a very high level. Perhaps most notably, linear transformations may be completed in one effective 'step' whereas nonlinear transformations require many 'steps'. Whether this is accurate or not in practice is somewhat dubious, but for our particular application it will indeed be.

This is relevant because we can get an idea of how to make an extremely fast (to run, that is) language model by considering what exactly happens during autoregressive inference. When one considers autoregressive inference, it is generally noted that models like Transformers that compare all tokens to all other tokens scale with $n^3d$ time complexity without caching, and $n^2d$ with caching for $n$ tokens and hidden dimension $d$. It is less appreciated that inference time also depends on the number of layers of linear transformations in a model $l$, as because typically each layer is separated from each next layer by one or more nonlinear transformation (layer normalization, ReLU, GeLU etc.) such that the actual time complexity becomes $n^2dl^2$ as each of the $n^2$ token comparisons require $l$ steps. 

Clearly it would be advantageous to reduce the number of layers in a model, but how can this be done while maintaining an efficiently trainable architecture? If a





