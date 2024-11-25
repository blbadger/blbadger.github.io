## Masked Mixers II

We continue exploring topics from [Part I](https://blbadger.github.io/smaller-lms.html), with a more thorough version of these results found in [this paper](https://arxiv.org/pdf/2409.01482)
Application of masked mixers to larger datasets will be explored, theory is expanded, and the nature of the learned structure of masked mixers versus transformers is investigated.

### Why Masked Mixers?

In [Part I](https://blbadger.github.io/smaller-lms.html) the motivation behind masked mixers for language modeling was detailed, but can be restated briefly: from high-dimensional projection theory (specifically the Johnson-Lindenstrauss lemma) we can accurately represent points in high-dimensional space in a space with much fewer dimensions (approximately the log of the number of points we have). This is an important result here because it implies that one can make language models that are capable of fitting extremely large datasets with what is currently considered an extremly small number of parameters: for example, a dataset one million times the size of the already large 15 trillion token Llama-3.1 dataset ($1.5 * 10^{19}$ tokens to be precise) may be arbitrarily well fit by a 352-dimensional model, which would require around 100 million parameters for a llama-style generative model.

Empirically it appears that the current state-of-the-art model type (the transformer) trains much too slowly to achieve this feat in any reasonable length of time, and the best-performing models are typically trained on thousands of GPUs for many days. From other investigations on information transfer between model layers, we wondered whether an architecture that more accurately represents its inputs (the masked mixer) would learn more efficiently.

Restricting our investigations to a small dataset (TinyStories, ie short children's stories written by ChatGPT) we found that although masked mixers are more efficient learners than the original GPT-style transformers for the task of causal language modeling, highly optimized and current transformers learn even more efficiently.  On the other hand, masked mixers were found to be much more efficient for language retrieval which is expected from their superior input representation properties.

### Accuracy and Flexibility

Even with these promising features, the question may be asked: why masked mixers? Before proceeding to further investigations on this architecture, it is worth considering what you get when you swap self-attention for masked convolutions.

Before masked mixers were first trained, it was found that these models give much more accurate representations of their inputs than transformers. In effect, this means that given an input, the information necessary to uniquely identify that input via gradient descent is retained throughout the mixer but not the transformer. It could be argued for or against the idea that this would be useful for language generation, and perhaps more convincingly argued that this is important for langauge retrieval. But accurate input representation remains a useful feature of mixers for a variety of applications. 

Perhaps as important is the model architecture *flexibility* of masked mixers. From investigations into representation in [vision transformers](https://blbadger.github.io/vision-transformers.html), it was apparent that vision transformers require each key component of their architectures: layer normalizations, MLPs, self-attention, and residual connections are all required for effective gradient propegation.

This can be tested more directly by simply removing each component and observing the training process. Ironically for an architecture introduced as 'Attention is all you need', self-attention is actually the only removable component (as long as it is replaced by some other trainiable inter-token parameters): removal of MLPs, layer norms, or residual connections results in very poor language model training with either a failure to minimize a loss function (even if MLPs are removed or replaced with attention) or else training becomes unstable (for removal of layer norms or residuals) and gradients spikes to infinity. The reason for this is that attention is a rather difficult transformation for gradients to propegate across, and this is important because it essentially fixes the architectures of models with attention to similar patterns, all requiring some form or other of the same components transformers have (layer norms, residuals, MLPs, ets.).

On the other hand it turns out that langauge training proceeds perfectly well but is slightly less efficient when layer norms are removed from the masked mixer architecture. Even on a relatively large and difficult dataset such as the `fineweb-edu 10BT`, a 16-layer masked mixer with no layer norms whatsoever does not experience any spikes in loss during training, provided that the learning rate is relatively modest (<20%).

It is interesting to note that this is not the case for residual removal.

This means that the mixer is effectively much more flexible than the transformer, and can be modified to a much greater extent. This topic will be explored more in the 'Linear mixer' section of this page.

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

These are very large performance gaps: recall that the difference between transformer and mixer CLM loss is typically 0.5-2%, such that with a modest increase in training duration one architecture is able to achieve the loss of the other. But from the figure below it is apparent that it would take a huge number of steps (perhaps 1000x) for the transformer to match the mixer's loss achieved, if it ever is. The figure below provides the loss curves upon various training runs. Note that the 1024-dim mixer is more or less equivalent in memory and somewhat faster than the 512-dim transformer model, and that the mixers are trained with dropout ($p=0.05$) hence the drop in evaluation loss compared to training loss at all steps.

![autoencoders](/deep-learning/language_autoencoders.png)

The gap is even larger when we consider that the mixer occupies a much smaller memory footprint for identical $d_m, n_l$ parameters. If we match the mixer to the $d_m=1024, n_l=4$ transformer's memory on device by doubling the $n_l \to 8$, the mixer reaches 1.65/1.37 train/test loss using the same compute (4x V100s, 6h) as the above transformer. This would be expected to require hundreds or thousands (!) of epochs for the transformer to match, and in that way one could claim that the mixer is hundreds or thousands of times as efficient an autoencoder as a transformer.

### Fineweb Modeling Efficiency

The goal of a machine learning algorithm is to minimize some loss function on a dataset efficiently, and the hope is that the minimization process and dataset are sufficient to generalize to the task you actually want to perform (typically representation by a 'test' or 'evaluation' dataset). The choice of a loss function, the model architecture to use, the optimization approach, the amount of compute employed, and the dataset are all important factors in whether the generalization actually occurs.

In [Part I](https://blbadger.github.io/smaller-lms.html) this question was addressed for two model architectures on a relatively small language dataset composed of synthetic one-paragraph children's stories written in the vocabulary of a four-year-old. There it was found that masked mixers are nearly as efficient language modelers as transformers for next token generative tasks, and far more efficient retrieval models.

It may be wondered just how generally applicable these results are: perhaps mixers with their relatively simple inter-token linear transformations are effective modelers of TinyStories because that dataset is itself extremely simple? If this were the case then one would expect to find that the masked mixer is much worse than modern, optimized transformers for causal language modeling on more complex datasets.

This hypothesis must be taken seriously because lack of correspondence of model training from a very small and self-contained dataset to larger ones is somewhat common in the deep learning field. Examples abound of model architectures that effectively modeled small datasets but were found to be relatively inefficient for large ones. Notable vision model cases are the variational autoencoder, which models MNIST well but is not powerful enough to train efficiently on ImageNet, and the Generative Adversarial Networks that model small and medium-sized datsets with some training instability but suffer from more frequent training instabilities upon application to larger and more varied datasets. As a counterpoint to those examples, it should be noted that GANs (and to some extent VAEs) are generally more efficient and flexible modelers of very small datasets (MNIST etc.) compared to diffusion models, but the latter have proven to be much more effective for large datasets.

Applying masked mixers and modern transformers to larger, more difficult datasets with more compute can give us an indication whether the masked mixers would or would not be efficient general language learners. The [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset is particularly well-suited to this kind of test, as it is an extensively curated dataset containing a wide variety of text that has been shown to be capable of training large language models more efficiently than less-curated datasets. Specifically, this dataset began as a compilation of the Common Crawl, underwent multiple rounds of filtering via LLMs for quality and then educational content and finally deduplication. This dataset is designed to be similar but somewhat more efficient than (proprietary) datasets used to train Mistral and Llama models, and can in that respect be considered to be much more difficult to model than TinyStories. We use the 10 billion token (GPT2 tokens, that is) subset of the `fineweb-edu` dataset so that our relatively small models may be trained in a reasonable amount of time on a single 4x V100 compute node.

The primary challenge of training a model on a large dataset like this (versus a small one) is that the larger the datset, the less likely it can be stored in memory for fast access during batched forward and reverse passes during training. To see why this is, observe that each token is usually stored as a `torch.long = torch.int64` datatype, meaning that each token requires eight bytes. A ten billion token dataset would therefore be expected to require around 80 GB of memory, and for distributed data parallel training each GPU requires its own dataset by default (although this could be modified if necessary). Thus we can expect to require 320 GB for a four-GPU system, which is currently a little more than the node used here contains.

One option would be to simply increase the existing server's memory (more than one terabyte of memory can be installed in this machine) but that is a temporary solution, as any dataset larger than this memory value will experience the same problem that we are meeting with the `fineweb-edu`.  Very large datasets may be streamed directly from storage in the cloud (an S3 bucked or Azure Blob, for example) such that a local machine never stores more than a fixed amount of an arbitrarily large dataset, but this approach is heavily bandwidth-dependent. In the author's experience streaming large datasets lead to poor GPU usage for training smaller models, where the forward and reverse passes do not provide enough time to load a batch of data over the network without subsequenty delays. Instead, we can use clever data loading algorithms to load training and test data from storage into memory and back again during the training process, a process analagous to streaming the data from disk to CPU memory, and thence to GPU memory. Modern solid state drives read and write contiguous data at speeds of ~500MB/s, which is much faster than one will typically see for network streaming (which does typically not match one's internet bandwidth).

With this approach settled on, there are a number of subsequent design considerations to make: for example, do we want to load only the text from storage and tokenize each batch before sending the tokens to the GPUs, or do we want to tokenize ahead of time and simply load the tokens directly and send these to GPUs? Some quick testing is enough to show that tokenizing large batches (say $n=128, n_{ctx}=1024$) leads to poor GPU allocation and delays in data processing, so we take the latter approach. We make use of the HuggingFace `datasets` library, which implements datasets as PyArrow (python bindings for the C++ Apache Arrow libs) tables. This library is handy for fast loading from disk, and was chosen as it is well-integrated with the `fineweb-edu` dataset schema without too many modifications for most tasks. 

`fineweb-edu 10BT` is composed of approximately 10 billion tokens distributed across around 90 million examples. In the following code snippet, we use batch encoding to store the first 1024 tokens of each of these examples (padding where necessary) using the `datset.map()` functionality.

Let's examine a random sample of the `fineweb-edu 10BT` dataset. This dataset is stored as an Arrow table, and upon calling the `datasets.load_from_disk()` or `load_dataset` utility each row of this table may be represented Python dictionary. Below is one such sample (the text is truncated for brevity). Here we can see the `'text'` itself, a universally unique identifier for this sample (`'id'`), the actual source `'url'`, the Common Crawl dump source `'dump'`, and the path to the file's cloud storage (`'file_path'`), the `'language'`, a number of measurements determining the quality and educational content of this text (`'language_score', 'score', 'int_score'`), and the number of (GPT2) tokens in the text.

```python
{
'text':['Researchers from the United States and Switzerland have developed mathematical and statistical tools for reconstructing viral populations using pyrosequencing, a novel and effective technique for sequencing DNA. They describe their findings in an article published May 9th in the open-access journal PLoS Computational Biology.\nThe scientists knew that pyrosequencing reads are short and error-prone, and thus set out to improve upon this process...'],
'id': ['<urn:uuid:b3a05e48-160f-424f-8545-119be2db0560>'],
'dump': ['CC-MAIN-2017-26'],
'url': ['https://www.eurekalert.org/pub_releases/2008-05/plos-ncm050808.php'],
'file_path': ['s3://commoncrawl/crawl-data/CC-MAIN-2017-26/segments/1498128320270.12/warc/CC-MAIN-20170624170517-20170624190517-00112.warc.gz'],
'language': ['en'],
'language_score': [0.846785843372345],
'token_count': [609],
'score': [2.71875],
'int_score': [3]
}
```

We want to tokenize the text, but don't necessarily care about the metadata. Perhaps the fastest approach is to tokenize the text in batches, truncating samples that are too long and padding those that are too short as follows

```python
def tokenization(example):
    tokens = tokenizer.batch_encode_plus(
		example['text'],
		add_special_tokens=False,
		return_tensors='pt',
		truncation=True,
		max_length=1024,
		padding='max_length',
		padding_side='right'
             )
    return tokens
```

where we use the dictionary lookup `example['text']` to access the text. We can then perform a train/text split of the dataset and add the tokens to the dataset via batched `dataset.map()` with the tokenizer above, and save the resulting tokenized dataset to disk in the desired location.

```python
import datasets
from datasets import load_dataset, load_from_disk

def tokenization(example):
	...

def map_dataset(text, path):
	dataset = text.map(tokenization, batched=True)
	dataset.save_to_disk(train_path)
	return

if __name__ == '__main__':
	train_path = "/path/to/store/training/tokens"
	train_path = "/path/to/store/test/tokens"
	dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT", streaming=False)
	train_data, test_data = dataset.skip(split_index), dataset.take(split_index)
	map_dataset(train_text, train_path)
	map_dataset(test_text, test_path)
```

If desired, one can remove the unneeded keys from each dataset entry (columns in Arrow format) by using a `dataset.map()` that iterates through each example's keys, deleting all that are not requried.

The method above works well for higher-context (512 and 1024) entries but leads to too many tokens being removed when smaller context windows (<512) are required. In this case, we cannot use batched tokenization examples are typically of different lengths: instead we perform unbatched tokenization without truncation followed by reshaping such that each sample has `len(tokens) // n_ctx` tensors each with `n_ctx` tokens. What we are doing here is to split up each example's sequence of tokens into a batch of many sequences tokens, discarding the last sequence if its length is less than `n_ctx` (which is faster than padding). The following example is called during `map_dataset()` via `train_dataset = train_text.map(tokenization, batched=False)` and the same for the test dataset.

```python
def tokenization(example, n_ctx=32):
    tokens = tokenizer.encode_plus(
			example['text'],
			add_special_tokens=False,
			return_tensors='pt',
			truncation=False,
			padding=False,
		).input_ids
    tokens = torch.flatten(tokens, start_dim=0)
    batch_size = len(tokens) // n_ctx 
    length = n_ctx * batch_size
    tokens = tokens[:length].reshape(batch_size, n_ctx)
    return {'input_ids': tokens}
```

Debatching the input is trickier than it sounds: we really want to convert each example into however many examples there are batches in that sample, which the `datasets` library does not natively support. Instead we can form an array of inputs (one for each batch sample in our example), convert the array to a PyArrow Table object and return that object to the mapper. The key to this approach is that `dataset.map()` only allows batched outputs if `batched=True` is specified in the mapper args, but we actually need a `batch_size=1` (unbatched) input as each example is expected to have a different batch size than any other example. This can be implemented as follows: after the dataset is tokenized into batches as above, it is loaded and debatched and saved to a new dataset object as open datasets cannot be overwritten.

```python
def debatch(example):
	batch_size = len(example['input_ids'])
	keys = list(example.keys())
	debatched_inputs = [{'input_ids': tokens} for tokens in example["input_ids"][0]]
	examples = pa.Table.from_pylist(debatched_inputs)
	return examples

test_dataset = test_dataset.map(debatch, batched=True, batch_size=1)
train_dataset.save_to_disk(train_path + '-debatched')
```

Making use of `datasets` PyArrow objects is straightforward: first we use `load_from_disk` and then the datsets may be passed directly to the `transformer.Trainer` as follows:

```python
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)
...
trainer = transformers.Trainer(
	model=model,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
```

Once this is done, we can observe model performance on the `fineweb-edu`! Some quick testing tells us some things that one would probably expect: firstly that this more difficult dataset requires much deeper models than `TinyStories`, as the latter was most efficiently modeled by 4- or 8-layer mixers and transformers whereas the `fineweb-edu` is most efficiently modeled by 16 (or more) -layer models. We mostly stick to 16-layer models, as these are the deepest that for reasonable $d_m$ layer widths (batch size of 128 or larger) such that `fineweb-edu 10BT` is trainable in under one day on a 4x V100 node. We find that $d_m=1024, n_l=16$ mixers train in approximately the same time and memory per step as $d_m=512, n_l=16$ Llama-style transformers (4-headed) that are otherwise optimized for minimum loss per given compute. 

Loss curves during masked mixer and transformer training on `fineweb-edu 10BT` are given below, where each training run requires approximately 20 hours on the 4x V100 (batch sizes are modified to maximize memory usage such that $n_{ctx}=32$ samples are trained with a total batch size of $4* 512 = 2048$, $n_{ctx}=128$ with batches of size $4* 128 = 512$, and $n_{ctx}=512$ with batches of size $4*32=128$ etc).

![fineweb_loss](/deep-learning/fineweb_clm_loss.png)

There are a few notable observations from the above figure, the primary being the answer to the question posed at the start of this section that masked mixers scale just as well (if not better than transformers) to large and difficult datasets. Recall that the loss gap between the most-optimized Llama model and masked mixer for language generation evaluation was 1.71 and 1.61 (6%) for $n_{ctx}=512$. For the same $n_{ctx}$, we see a smaller gap (less than 3%) corresponding to fewer extra training steps required to have the masked mixer's accuracy match that of the transformer. Furthermore, for $n_{ctx}=512$ it is apparent that the gap between mixer and transformer loss narrows as training proceeds such that with more compute applied one would expect the masked mixer to be the more efficient architecture given the current constraints. This notion is supported by observing that $n_{ctx}=128$ models experience exactly this phenomenon of the course of their training, which uses 4x the batch size and thus 4x the number of samples that $n_{ctx=512}$ training does.

From earlier investigations we hypothesized that the transformer is a somewhat more efficient causal language model than the masked mixer because next token prediction is fundamentally a many-to-one mapping: we have many previous tokens and only get logits for one next token. This mapping mirrors the intrinsic properties of the transformer's attention operation, where information from most inputs is removed during the forward pass. If this were the case then one would expect for next token prediction with fewer previous tokens (which is closer to a bijective mapping) to favor the masked mixer, and this is precisely what is found for $n_{ctx}=32$ or even $n_{ctx}=128$.

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

When we compare mixer versus transformer performance on bidirectional token prediction versus causal language modeling-style next token prediction on the `fineweb-edu` 10BT dataset, we see that the relative performance is nearly identical, suggesting that it is not language stochasticity but many-to-one mapping that give transformers a training efficiency advantage over masked mixers for token prediction.

![uni vs bidirectional](/deep-learning/uni_vs_bidirectional.png)


### One Step Language Completion Efficiency

So far we have seen that masked mixers are better at tasks requiring approximately bijective functions like autoencoding or retrieval, and worse at tasks requiring injective mappings such as causal language modeling (where many previous tokens are mapped to one next token). It could be wondered how efficiently each model learns a task that exhibits aspects of both injective and bijective mappings, say one-step text completion on a per-token basis. The hypothesis is that these models will be approximately equivalently efficient, assuming that this task requires a relatively even mix of bijective and injective mappings

To elaborate, the task is to generate all the completion tokens in a text segment in one step. Arbitrarily choosing text segments of length 512 such that the model recieves information from tokens ($\{t_0, t_1, .., t_{255}\}$) and generates tokens $\{t_{256}, t_{257}, ... t_{511}\}$, we want information from each of the first 256 tokens to be available for the subsequent 256 tokens during the forward pass. One way to do this would be to use the last hidden layer of token $t_{255}$ as an input to a decoder in a similar architecture to the autoencoders presented above, but this would require all input information to be present in that token and we have already seen that this sort of task is much better suited for masked mixers. Instead we want to provide information from all input tokens $\{t_{0-255}\}$ directly, while preventing the passage for information from the prediction tokens $\{t_{256-511}\}$.

Perhaps the most efficient way to accomplish this would be to change the causal language masking from lower triangular to block form (where the weights from all prediction tokens are masked, and weights from input to outputs are bidirectional). This would require substantial changes to the causal language masking implementation for transformers, however, but there is a much simpler method: retain causal langauge masking but mask the hidden layers of prediction tokens at a certain layer of the model. We use an encoder-decoder model in which the masking occurs half-way through the models layers (at layer 8 or a 16-layer model to be precise) paired with loss masking on the inputs. This can easily be applied to both masked mixers and llama-style transformers, and for clarity here is a depiction of this process:

![fineweb_loss](/deep-learning/completion_model.png)

As expected, there is nearly complete parity in training efficiency for the masked mixer and transformer when applied to this one-step completion paradigm. Note that both train relatively poorly: unsurprisingly, it turns out that attempting to predict many tokens at once in one step is much more difficult than predicting one token at a time.

![fineweb_loss](/deep-learning/mixer_vs_llamacompletion.png)

### Linear Mixers

In the field of numerical analysis one can generally say that there are a number of differences between linear and nonlinear processes, at least at a very high level. Perhaps most notably, linear transformations may be completed in one effective 'step' whereas nonlinear transformations require many 'steps'. Whether this is accurate or not in practice is somewhat dubious, but for our particular application it will indeed be.

This is relevant because we can get an idea of how to make an extremely fast (to run, that is) language model by considering what exactly happens during autoregressive inference. When one considers autoregressive inference, it is generally noted that models like Transformers that compare all tokens to all other tokens scale with $n^3d$ time complexity without caching, and $n^2d$ with caching for $n$ tokens and hidden dimension $d$. It is less appreciated that inference time also depends on the number of layers of linear transformations in a model $l$, as because typically each layer is separated from each next layer by one or more nonlinear transformation (layer normalization, ReLU, GeLU etc.) such that the actual time complexity becomes $n^2dl^2$ as each of the $n^2$ token comparisons require $l$ steps. 

Clearly it would be advantageous to reduce the number of layers in a model, but how can this be done while maintaining an efficiently trainable architecture? To start to answer this question, one can instead ask the following: ignoring trainability, what is the minimum number of layers in a causal language model? The answer is straightforward: a one-layer model contains the fewest layers, and a one-layer model in this case is equivalently a fully linear model. The question then becomes whether or not a linear or nearly-linear model is capable of being trained effectively, either on a small dataset like TinyStories or a larger dataset like the Fineweb.

For small datasets like TinyStories, the answer is somewhat surprisingly yes: fully linear language models are capable of generating gramatically correct text, and nearly-linear models are only slightly less efficient to train than highly non-linear, many-layered models explored previously. 

Before proceeding further, however, it is best to understand a few theoretical arguments for and against the use of linear models. The arguments for mostly revolve around their utility: they are fast (because they can be mostly or wholly parallelized on hardware), easy to optimize, and somewhat more interpretable than nonlinear ones. The downsides revolve around their lack of representational power: 

How might one go about converting a masked mixer into a linear model? We will take the approach to remove nonlinear transformations and optimize the resulting linear model to train most efficiently using adaptive versions of stochastic gradient descent.

The equation for layer normalization is as follows: for a specific layer input $x$, the output $y$ is

$$
y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} * \gamma + \beta
$$

where $\gamma$ and $\beta$ are trainable parameters (which are usually held in very high precision) and $\mu, \sigma^2$ denote the mean (expectation) and variance of the random variable $x$. This is clearly a nonlinear operation (although it could be transformed into a linear one) and so we will simply remove this transformation for now.




