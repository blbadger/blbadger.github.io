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

[loss figure here]

Why would causal masking be so important to a model that does not perform causal modeling? It may be presumed that causal masking helps training efficiency because 


### Text Compression

Although it may not seem to be very important to the field of artificial intelligence, text compression in many ways has been proven time and time again to be a very apt predictor of a language model's abilities in areas usually lumped together in that field, and has been shown to be important for everything from language generation to q/a chat capability to chain-of-thought reasoning capability. 

Language compression is an old goal, and attempts to understand how to compress text and how much text can be compressed go back to the beginnings of information theory. Shannon's [seminal work](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) focuses on the problem of compression and transmission of information of text,

One of the most general methods of measuring text compression is bits per byte (bpb), which is the number of bits required for encoding a byte of input text. Often the input is assumed to be encoded in UTF-8, which uses one byte per character and makes this measure effectively the number of bits per input character.

Although less well known than other model capabilities, the most effective text compression methods today are frontier large language models trained to predict each next token in a very large corpus of text. The gap between classical compression algorithms and these models is vast in the scale of information theory: perhaps the most widely used compression algorithm `gzip` achieves up to around 2 bits per byte, highly tuned dictionary and bit matcher decoders achieve around 1.2 bits per byte, whereas Deepseek v3 achieves 0.54 bits per byte.

The way large language models compress text is simply by being able to predict next tokens, where the compression is simply the number of bits required to correct errors in the model's prediction for each token.  Causal language model-style compression is nearly as old as text compression itself. For example, Shannon used next character prediction by people as a way to attempt to estimate the source entropy for the English language. Shannon estimated a lower bound of around 0.6 bits per character, very similar to what we see for large language models today.

There is a clear upper bound to causal language model text compression, however: in natural languages such as English, there is a certain amount of inherent ambiguity such that no particular word necessarily follows from a given previous sequence of words. This is what Shannon referred to as 'source entropy', and it may be thought of as irreducible error and provides a hard lower bound on the bits-per-byte compression of a causal-style model. 

With this in mind, we have a clear alternative to next token prediction-based compression. We can use our new masked mixer-based autoencoder to compress the input directly and thereby avoid the problem of source entropy alltogether. The reason for this is that our autoencoder effectively compresses the entire input into one vector and uses this vector to reconstruct that input, where the source entropy becomes irrelevant for an arbitrarily powerful autoencoder capable of capturing all necessary information in the embedding vector. In real-world applications the source entropy is clearly important to the ease of actually training a model (as we shall see for mathematical versus general text later), but in the idealized scenario of arbitrary power our only lower bound is the size of the autoencoder's embedding.


### Embedding-augmented causal language models



### Langauge Model Memory



