## Language Mixers IV: Memory Models

This page focuses on information compression, specifically how we can achieve better compression via new language model architectures and how this can be used for large-context models.

### Introduction

In work detailed in [this blog post](https://blbadger.github.io/smaller-lms-2.html) and [this paper](https://arxiv.org/pdf/2409.01482), a major finding was that one can modify a new causal language model architecture termed the 'masked mixer' (which is essentially a transformer with self-attention replaced by masked 1D convolutions) to effectively autoencode inputs with high compression, that is, one can train small models in reasonable time to be able to regenerate (with some error) a sequence of 512 or more tokens using the embedding of only one token with excellent generalization properties. It was found that using masked mixers for encoder and decoder allows for far greater input autoencoding accuracy than a transformer encoder/decoder pair for a given model size, leading to the ability to compress inputs in new and potentially more efficient methods than is possible using the transformer.

It may be wondered why text compression ability is important: even if large language models achieve 2x or 4x the compression of commonly used algorithms like `gzip`, they are thousands of times more computationally expensive to actually run and thus not preferred today (although this may change in the future). The answer is that effective compression from many to one token allows one to design architectures that have interesting properties such as extended context windows or meta-context guidance.

We will first tweak the autoencoder mixer architecture to try to obtain optimal text compression in fixed compute budgets before using this information to attempt to test whether one can obtain better text compression ratios than the current best methods (ie large transformers). We will conclude by examining the suitability of autoencoding embeddings for extending generative language model context with sub-quadratic complexity.

### Mixer autoencoder optimizations

To begin with, it is helpful to recall the architecture of the masked mixer-based autoencoder as presented in the work linked in the last section:

![mixer autoencoder architecture](/deep-learning/mixer_autoencoder.png)

This architecture recieved little to no optimization in that work, as it was mostly presented as evidence for a hypothesis involving bijective functions and autoencoding. But now that we want to improve and elaborate upon this architecture, where would we start? 

One obvious question is whether or not the convolutions really need to be masked: after all, we are generating the output in one step and are not adhering to the causal language modeling objective of next token prediction, so why would we really want to mask the model's convolutions? Bearing this in mind, it is somewhat unexpected that removing the encoder convolution's mask results in substantially less efficient training and even more surprising that removal of the decoder's mask (keeping the encoder unmasked as well) results in highly unstable training with gradient norm spikes leading to rapid rises in loss during training.

![mixer autoencoder efficiencies](/deep-learning/autoencoder_causality.png)

Why would causal masking be so important to a model that does not perform causal modeling? There is usually some benefit to matching a model's structure to any prior we know about the dataset that is being modeled, and with that perspective one could perhaps guess that enforcing causality is beneficial because the data being modeled (text) is in some way fundamentally causal as it is understood in one orientation. It is less certain why removing all causality masks leads to highly unstable training, as one may expect for a simple decrease in efficiency in this paradigm rather than exploding gradients. 

In addition to whether encoders and decoders should be causally masked, it may also be wondered whether the masked mixer is even the optimal architecture for the decoder, and in particular whether a transformer decoder might be more effective. In the [paper](https://arxiv.org/pdf/2409.01482v1) introducing the mixer autoencoder, it was observed that transformer-based autoencoders are far less efficiently trainable than autoencoders based on the masked mixer. There are fairly convincing theoretical and empirical arguments to suggest that this is primarily because self-attention is simply not well suited for encoding if one cares about capturing as much of the input as possible. This does not preclude the possibility that transformers may be well suited for the autoencoder's decoder, however. 

From an architectural perspective, the masked mixer architecture is largely based on the transformer such that we are essentially testing whether or not a masked convolution (with a somewhat larger $d_m$ in the decoder) is more efficient to train than multi-headed attention (with a somewhat smaller $d_m$ due to memory and FLOPs required to compute the $K, Q, V$ projections). This can be appreciated when we compare the autoencoder architecture diagram above with the transformer-mixer hybrid autoencoder below:

![mixer autoencoder efficiencies](/deep-learning/autoencoder_transfixer.png)

Recalling that transformers require approximately double the device (GPU) memory for the same $n_l, d_m$ configuration as masked mixers due to the large number of activations formed by $K, Q, V$ projections, we can compare a transformer decoder of dim $d_m/2$ to a masked mixer decoder of dim $d_m$ all else equal.

![decoder options](/deep-learning/autoencoder_decoder_fig.png)

### Text Compression Background

Although it may not seem to be very important to the field of artificial intelligence, text compression in many ways has been proven time and time again to be a very apt predictor of a language model's abilities in areas usually lumped together in that field, and has been shown to be important for everything from language generation to q/a chat capability to chain-of-thought reasoning capability. 

Language compression is an old goal, and attempts to understand how to compress text and how much text can be compressed go back to the beginnings of information theory. Shannon's [seminal work](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) focuses on the problem of compression and transmission of information of text,

One of the most general methods of measuring text compression is bits per byte (bpb), which is the number of bits required for encoding a byte of input text. Often the input is assumed to be encoded in UTF-8, which uses one byte per character and makes this measure effectively the number of bits per input character.

Although less well known than other model capabilities, the most effective text compression methods today are frontier large language models trained to predict each next token in a very large corpus of text. The gap between classical compression algorithms and these models is vast in the scale of information theory: perhaps the most widely used compression algorithm `gzip` achieves up to around 2 bits per byte, highly tuned dictionary and bit matcher decoders achieve around 1.2 bits per byte, whereas Deepseek v3 achieves 0.54 bits per byte.

The way large language models compress text is simply by being able to predict next tokens, where the compression is simply the number of bits required to correct errors in the model's prediction for each token.  Causal language model-style compression is nearly as old as text compression itself. For example, Shannon used next character prediction by people as a way to attempt to estimate the source entropy for the English language. Shannon estimated a lower bound of around 0.6 bits per character, very similar to what we see for large language models today.

There is a clear upper bound to causal language model text compression, however: in natural languages such as English, there is a certain amount of inherent ambiguity such that no particular word necessarily follows from a given previous sequence of words. This is what Shannon referred to as 'source entropy', and it may be thought of as irreducible error and provides a hard lower bound on the bits-per-byte compression of a causal-style model. 

With this in mind, we have a clear alternative to next token prediction-based compression. We can use our new masked mixer-based autoencoder to compress the input directly and thereby avoid the problem of source entropy alltogether. The reason for this is that our autoencoder effectively compresses the entire input into one vector and uses this vector to reconstruct that input, where the source entropy becomes irrelevant for an arbitrarily powerful autoencoder capable of capturing all necessary information in the embedding vector. In real-world applications the source entropy is clearly important to the ease of actually training a model (as we shall see for mathematical versus general text later), but in the idealized scenario of arbitrary power our only lower bound is the size of the autoencoder's embedding.

### Text Compression in Masked Mixer Autoencoders

If we have a negative log likelihood loss $\Bbb L$, we can compute the number of bits per input byte for a given segment of text if we know the length of text in bytes $L_b$ and the number of tokens required to encode that text for a given tokenizer $L_t$. 

$$
\mathrm{bpb} = (L_t / L_b) \Bbb L / \ln (2)
$$

On this page we report loss as the `torch` implementation of `CrossEntropyLoss`, which is equivalent to taking the Negative Log Likelihood of a softmax-tranformed logit output. This means that we can simply substitute our CEL loss values for the negative log likelihood $\Bbb L$ values (the softmax simply transforms the model's outputs to a probability distribution). We also make the simplifying assumption that our text is encoded in single-byte UTF-8.

### Embedding-augmented causal language models

Recalling our original motivation to introduce input embeddings to remove the source entropy of text for a language model compressor, it may be wondered if one cannot combine the advantages of the autoencoder with next token prediction-based compression. The reson why this might be beneficial is as follows: it is apparent that masked mixer autoencoders require a compressed $d_m$ that is too large (in bits per $n_{ctx}$ tokens) to improve upon the compression found using next token prediction models given the relatively small amount of compute we have been applying to train these models. 

The primary reason for this is that each token (which usually representes between 3 and 5 bytes of text) is greatly expanded by the embedding transformation, whereby each token becomes represented by vectors of at least 512 elements, or at least 1024 bytes in fp16. This is such a large expansion that even our subsequent many-to-one token compression abilities do not give greater total compression than causal language models.

Some reflection may convince us that there is a good reason for this: it may be conjectured that our autoencoder less efficiently trainable than a next-token-prediction model (for our compute) because it must generate the entire context window in one forward pass, whereas the causal language model generates one next token at a time using information from all previous tokens. 

With this in mind, it may be wondered if we can combine the advantages of causal modeling and autoencoding to make an encoder-decoder architecture that exhibits superior compression ability to either autoencoder or causal language model alone, give similar compute to what has been used previously.

There are a few options for 

![memory decoder architectures](/deep-learning/memory_encoder_options.png)

![memory decoder performances](/deep-learning/decoder_options.png)

Can adding an encoder's embedding lead to increased compression? To answer this we first need to know how large our embeddings are (particularly how many bytes they require) and then we can convert this to a bits-per-byte value.  Suppose one trains an embedding-augmented causal model where the embedding is of dimension $n_p$, each parameter being stored using $b_p$ bits, for a context window of size $n_{ctx}$ and $L_b / L_t$ bytes of input text per token. Then we can calculate the bits per byte required to store this embedding (amortized over the input) as follows:

$$
\mathrm{bpb} = \frac{n_p * b_p} / \frac{n_{ctx} * (L_b / L_t)}
$$

Once this value is known, we can find the loss offset $\Bbb L_o$ that corresponds to this extra required information,

$$
\Bbb L_o = \frac{\mathrm{bpb} * \ln 2}{(L_t / L_b)} 
$$

and add this offset to the causal language modeling negative log likelihood loss for next token prediction to find the total loss.

$$
\Bbb L = \Bbb L(O(a, \theta), y) + \Bbb L_o
$$

We call this the 'normalized loss' for brevity. For a relativel small embedding ($d_m=64$) assuming 4 bits per parameter, and with a context window of $n_{ctx}=1024$ we have the following normalized loss:

![memory decoder performances](/deep-learning/memory_compression_fig.png)

There is some expected behavior here: the embedding-augmented models begin training with higher loss (the result of the offset required for storage of the 64 floats in the embedding vector) but then approach or pass the purely causal model's loss (or equivalently its bpb compression of the input). 

It is interesting however that the masked mixer decoder appears to be able to learn to use the information present in the embedding more efficiently than the transformer decoder.

### Memory Models

The ability to compress information from a sequence of tokens into one embedding in an efficient manner has another utility: we can use these embeddings to provide exended context to a model without increasing its inference computation.

The architecture we will experiment with here is mostly similar to the embedding-augmented causal langauge model architecture implemented above, where we use the token dimension concatenation to maximize the number of embeddings we can provide to the decoder model. One notable difference

![memory decoder architectures](/deep-learning/memory_transformer.png)



