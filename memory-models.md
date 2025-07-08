## Language Mixers IV: Memory Models

This page focuses on information compression, specifically how we can achieve better compression via new language model architectures and how this can be used for large-context models.

### Introduction

In work detailed in [this blog post](https://blbadger.github.io/smaller-lms-2.html) and [this paper](https://arxiv.org/pdf/2409.01482), a major finding was that one can modify a new causal language model architecture termed the 'masked mixer' (which is essentially a transformer with self-attention replaced by masked 1D convolutions) to effectively autoencode inputs with high compression, that is, one can train small models in reasonable time to be able to regenerate (with some error) a sequence of 512 or more tokens using the embedding of only one token with excellent generalization properties. It was found that using masked mixers for encoder and decoder allows for far greater input autoencoding accuracy than a transformer encoder/decoder pair for a given model size, leading to the ability to compress inputs in new and potentially more efficient methods than is possible using the transformer.

It may be wondered why text compression ability is important: even if large language models achieve 2x or 4x the compression of commonly used algorithms like `gzip`, they are thousands of times more computationally expensive to actually run and thus not preferred today (although this may change in the future). The answer is that effective compression from many to one token allows one to design architectures that have interesting properties such as extended context windows or meta-context guidance.

We will first tweak the autoencoder mixer architecture to try to obtain optimal text compression in fixed compute budgets before using this information to attempt to test whether one can obtain better text compression ratios than the current best methods (ie large transformers). We will conclude by examining the suitability of autoencoding embeddings for extending generative language model context with sub-quadratic complexity.

### Mixer causality is non-optimial for encoders but necessary for decoders



### Text Compression



### Embedding-augmented causal language models



### Langauge Model Memory



