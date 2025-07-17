## Language Mixers IV: Memory Models

This page focuses on information compression, specifically how we can achieve better compression via new language model architectures and how this can be used for large-context models.

### Introduction

In work detailed in [this blog post](https://blbadger.github.io/smaller-lms-2.html) and [this paper](https://arxiv.org/pdf/2409.01482), a major finding was that one can modify a new causal language model architecture termed the 'masked mixer' (which is essentially a transformer with self-attention replaced by masked 1D convolutions) to effectively autoencode inputs with high compression, that is, one can train small models in reasonable time to be able to regenerate (with some error) a sequence of 512 or more tokens using the embedding of only one token with excellent generalization properties. It was found that using masked mixers for encoder and decoder allows for far greater input autoencoding accuracy than a transformer encoder/decoder pair for a given model size ad compute budget during training, leading to the ability to compress inputs using new and potentially more efficient methods than what has been possible using the transformer.

It may be wondered why text compression ability is important: even if large language models achieve 2x or 4x the compression of commonly used algorithms like `gzip`, they are thousands of times more computationally expensive to actually run and thus not preferred today (although this may change in the future). The answer is that effective compression from many to one token allows one to design architectures that have interesting properties such as extended context windows or meta-context guidance, which we will investigate under the broad name of 'memory'.

We will first tweak the autoencoder mixer architecture to try to obtain optimal text compression in fixed compute budgets before using this information to attempt to test whether one can obtain better text compression ratios than the current best methods (ie large transformers). We will conclude by examining the suitability of autoencoding embeddings for extending generative language model context with sub-quadratic complexity using encoder embeddings.

### Causal masking increases autoencoder training efficiency

To begin with, it is helpful to recall the architecture of the masked mixer-based autoencoder as presented in the work linked in the last section:

![mixer autoencoder architecture](/deep-learning/mixer_autoencoder.png)

This architecture recieved little to no optimization in that work, as it was mostly presented as evidence for a hypothesis involving bijective functions and autoencoding. But now that we want to improve and elaborate upon this architecture, where would we start? 

One obvious question is whether or not the convolutions really need to be masked: after all, we are generating the output in one step and are not adhering to the causal language modeling objective of next token prediction, so why would we really want to mask the model's convolutions? Bearing this in mind, it is somewhat unexpected that removing the encoder convolution's mask results in substantially less efficient training and even more surprising that removal of the decoder's mask (keeping the encoder unmasked as well) results in highly unstable training with gradient norm spikes leading to rapid rises in loss during training. The following figure details the cross-entropy losses achieved by masked mixer-based autoencoders ($d_m=1024, n_{ctx}=512, n_l=8, b=128$) on the `FineWeb-edu (10BT)` dataset, where the evaluation is a hold-out sample from the corpora (which has been measured to contain <5% duplicate documents). Here 'masked' indicates a causal (left-to-right) mask implemented using a lower triangular mask on the 1D convolution weight matrix, and the right panel is simply an extended training run of the conditions in the left panel (omitting the unstable no-masked model).

![mixer autoencoder efficiencies](/deep-learning/autoencoder_causality.png)

Why would causal masking be so important to a model that does not perform causal modeling? There is usually some benefit to matching a model's structure to any prior we know about the dataset that is being modeled, and with that perspective one could perhaps guess that enforcing causality is beneficial because the data being modeled (text) is in some way fundamentally causal as it is understood in one orientation. It is less certain why removing all causality masks leads to highly unstable training, as one may expect for a simple decrease in efficiency in this paradigm rather than exploding gradients. 

### Masked mixers versus transformer autoencoder decoders

In addition to whether encoders and decoders should be causally masked, it may also be wondered whether the masked mixer is even the optimal architecture for the decoder at all, and in particular whether a transformer decoder might be more effective. In the [paper](https://arxiv.org/pdf/2409.01482v1) introducing the mixer autoencoder, it was observed that transformer-based autoencoders are far less efficiently trainable than autoencoders based on the masked mixer. There are fairly convincing theoretical and empirical arguments to suggest that this is primarily because self-attention is simply not well suited for encoding if one cares about capturing as much of the input as possible. This does not preclude the possibility that transformers may be well suited for the autoencoder's decoder, however, although there are similar arguments to be made for the better suitability of masked mixers to decoding when one wants to generate all output elements simultaneously as is performed here.

From an architectural perspective, the masked mixer architecture is largely based on the transformer such that we are essentially testing whether or not a masked convolution (with a somewhat larger $d_m$ in the decoder) is more efficient to train than multi-headed attention (with a somewhat smaller $d_m$ due to memory and FLOPs required to compute the $K, Q, V$ projections). This can be appreciated when we compare the autoencoder architecture diagram above with the transformer-mixer hybrid autoencoder below:

![mixer autoencoder efficiencies](/deep-learning/autoencoder_transfixer.png)

Recalling that transformers require approximately double the device (GPU) memory for the same $n_l, d_m$ configuration as masked mixers due to the large number of activations formed by $K, Q, V$ projections, we can compare a transformer decoder of dim $d_m/2$ to a masked mixer decoder of dim $d_m$ all else equal. In the following figure, we observe losses upon training $n_l=16, n_{ctx}=1024$ models on the FineWeb. The 1024-dim transformer decoder is wall-clock-matched to the other models, such that the training run for that model was shortened to make the total compute applied to each model approximately constant.

![decoder options](/deep-learning/autoencoder_decoder_fig.png)

From these loss values it is clear that the masked mixer decoder is a good deal more efficient to train than the transformer decoder given the repeat embeddings from a masked mixer encoder. Earlier this was implied to be the expected result, and this is because transformers have been [shown](https://arxiv.org/abs/2409.01482) to be less capable than mixers in retaining information from the input in the output. This information loss characteristic is generally beneficial to causal language modeling where one wants a model to predict one next token given a sequence of many previous tokens (many of which are irrelevant to that next prediction), but here we want all tokens to be predicted in one forward pass. One would expect for a model that is capable of better informational retention to be more efficient to train as a decoder in this setting, and indeed this is what we have found, albeit for a somewhat limited amount of compute applied.

### Multi-headed mixer autoencoders

If the mixer is a better autoencoder encoder and decoder in this paradigm (where we regenerate all tokens of a sequence in one pass), how might one improve the mixer's architecture for further training efficiency? One straightforward guess might be to increase the number of inter-token trainable parameters, and this increase may be achieved in a number of different ways (multiple convolutional kernels, expanded convolutions with nonlinearities, multi-headed masked mixers) but when a number of architectures were tested for causal language modeling the most performant among these was the multi-headed mixer. The linear algebraic transformations that occur in the multi-headed mixer for the case where we have $n_h=2$ heads and the total projection dim is greater than the hidden dimension may be illustrated as follows:

![multi-headed autoencoders](/deep-learning/multiheaded_convs.png)

We modify the original multi-headed mixer architecture for the case where there are $n_h=2$ heads and the projection dim is just $d_m / n_h$, and fix $n_l=8$ for both encoder and decoder, and replace the mixer autoencoder's masked convolutions with these multi-headed convolutions.

![multi-headed autoencoders](/deep-learning/multi_headed_autoencoder.png)

As long as we stick to the $d_m / n_h$ total projection dimension limitation, the number of inter-token parameters $p$ for a model with $n_l$ layers and $n_{ctx}$ context is 

$$
p = n_l (n_h * n_{ctx}^2 + 2d_m^2)
$$ 

whereas we have $p = n_l * n_{ctx}^2$ inter-token parameters for the 'flat' masked mixer. Therefore we have a linear increase in inter-token parameters as the number of heads in this scenario, with a constant factor for the addition of any head. To see how to arrive at this number, observe that each head has a corresponding convolution (with weight matrix of size $n_{ctx}^2$) and the input and output projections are unchanged as the number of heads increases, in particular the output projection is size $d_m^2$ and the input $d_m*d_m *n_h / n_h$, and each head contains one 1D convolution.

For the case where we keep the concatenated projection dimension to be equal to $d_m$ as above, we have a notable increase in autoencoder training efficiency (see the figure below) relative to the 'flat' masked mixer which has no projections and only a single 1D-convolution. From the results below, it is clear that increasing the number of heads leads to greater training efficiency, and the gap between autoencoders with four or more heads and the flat masked mixer architecture is substantial.

![decoder options](/deep-learning/mixer_heads_figure.png)

Interestingly, however, the multi-headed mixer autoencoder experiences instabilities late in training manifesting in very rapidly exploding gradients for models with one or two heads. As this was observed early in training for autoencoders without causal masks, one straightforward explanation would be a problem with the multi-headed mixer's masks. We causally mask the convolution in each head, and a quick test shows that the encoder and decoder modules are indeed causal. A relatively straighforward solution for this problem would be to add a layer or RMS normalization to the concatenated projections, or add residuals across head elements. It is at present unclear why models with fewer heads would be more unstable to train than models with more heads.

It is also noteworthy that there is such a large difference in training efficiencies for multi-headed versus flat masked mixers for autoencoders. One can estimate the increase in training efficiency by finding the number of samples required to reach a certain loss value, and in this case one requires more than 6x the number of samples for the flat masked mixer to approach the loss achieved by the 4- or 8-headed autoencoder at 200k steps. For comparison, the difference in causal language model loss per step between flat and multi-headed mixers is very small: the flat mixer requires only around 1.05x the number of samples to match the 4-headed mixer when trained on TinyStories. If we implement a causal masked mixer while keeping the projection dimension equal to $d_m/n_h$, we find that there is very little difference in per-step loss between this model and the flat masked mixer when trained on the FineWeb-edu (10BT) dataset.

[insert fig]

Finally, from the figure above one clearly reaches limited returns when expanding beyond four heads as there is a significant computational burden but very little efficiency benefit with an increase to 8 heads, and an efficiency detriment if we increase to 16 heads. It is curious that four heads are also optimal causal transformer models of similar size with respect to loss achieved per unit of compute applied during training for this dataset as well.

### Text Compression Background

Although it may not seem to be very important to the field of artificial intelligence, text compression in many ways has been proven time and time again to be a very apt predictor of a language model's abilities across the spectrum and has been shown to be important for everything from language generation to q/a chat capability to chain-of-thought reasoning capability. 

Language compression is an old goal, and attempts to understand how to compress text and how much text can be compressed go back to the beginnings of information theory. Shannon's [seminal work](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) focuses on the problem of compression and transmission of textual information, as does earlier work in the field from Hartley and Nyquist. There were practical reasons for that: it has long been appreciated that one needs to send much less information to regenerate a string of characters than to generate the sound of someone speaking those characters, so figuring out exactly how much information is required was and is an important problem to data transmission. 

We focus on compression rather than transmission, although one can view deep learning models themselves as being somewhat similar to noisy communication channels. One of the most general methods of measuring text compression is bits per byte (bpb), which is the number of bits required for encoding a byte of input text. Often the input is assumed to be encoded in UTF-8, which uses one byte per character and makes this measure effectively the number of bits per input character if single-byte encoding predominates.

Although less well known than other model capabilities, the most effective text compression methods today are frontier large language models trained to predict each next token in a very large corpus of text. The gap between classical compression algorithms and these models is vast in the scale of information theory: perhaps the most widely used compression algorithm `gzip` achieves up to around 2 bits per byte, highly tuned dictionary and bit matcher decoders achieve around 1.2 bits per byte, whereas Deepseek v3 achieves 0.54 bits per byte.

The way large language models are usually used to compress text is simply by being able to predict next tokens, where the compression is simply the number of bits required to correct errors in the model's prediction for each token.  Causal language model-style compression is nearly as old as text compression itself. For example, Shannon used next character prediction by people as a way to attempt to estimate the source entropy for the English language. Shannon estimated a lower bound of around 0.6 bits per character, very similar to what we see for large language models today.

There is a clear upper bound to causal language model text compression, however: in natural languages such as English, there is a certain amount of inherent ambiguity such that no particular word necessarily follows from a given previous sequence of words. This is what Shannon referred to as 'source entropy', and it may be thought of as irreducible error and provides a hard lower bound on the bits-per-byte compression of a causal-style model. 

With this in mind, we have a clear alternative to next token prediction-based compression. We can use our new masked mixer-based autoencoder to compress the input directly and thereby avoid the problem of source entropy alltogether. The reason for this is that our autoencoder effectively compresses the entire input into one vector and uses this vector to reconstruct that input, where the source entropy becomes irrelevant for an arbitrarily powerful autoencoder capable of capturing all necessary information in the embedding vector. In real-world applications the source entropy is clearly important to the ease of actually training a model (as we shall see for mathematical versus general text later), but in the idealized scenario of arbitrary power our only lower bound is the size of the autoencoder's embedding.

### Text Compression in Masked Mixer Autoencoders

If we have a negative log likelihood loss $\Bbb L$, we can compute the number of bits per input byte for a given segment of text if we know the length of text in bytes $L_b$ and the number of tokens required to encode that text for a given tokenizer $L_t$. 

$$
\mathtt{BPB} = (L_t / L_b) * \Bbb L / \ln (2)
$$

On this page we report loss as the `torch` implementation of `CrossEntropyLoss`, which is equivalent to taking the Negative Log Likelihood of a softmax-tranformed logit output. This means that we can simply substitute our CEL loss values for the negative log likelihood $\Bbb L$ values (the softmax simply transforms the model's outputs to a probability distribution). We also make the simplifying assumption that our text is encoded in single-byte UTF-8.

We can now compare the compression achieved using masked mixer autoencoders to that obtained when using next-token-prediction models. Taking the FineMath 4+ dataset and a tokenizer that averages 2.82 characters (which equals 2.82 bytes assuming single UTF-8) and a model with a 512-dimensional embedding with an $n_{ctx}=512$ stored using 4 bits per parameter, we can calculate the amortized BPB as follows:

$$
\mathtt{BPB} = \frac{n_p * b_p}{n_{ctx} * (L_b / L_t)} \\
\mathtt{BPB} = \frac{512 * 4}{512 * 2.82} \approx 1.42
$$

assuming that we have zero loss after training (we actually have around $\Bbb L=0.7$). This compares disfavorably with the compression achieved by a causal language model transformer on this dataset using approximately the same compute,

$$
\mathtt{BPB} = (L_t / L_b) * \Bbb L / \ln (2) \\
\mathtt{BPB} = (1/2.82) * 1.4 / \ln(2) \approx 0.72
$$

A straightforward way to attempt to increase the compression in our autoencoder is to use a smaller embedding. Experimentally this is more efficient of one uses a larger $d_m$ in the encoder and decoder with a compression module that transforms $d_m \to d_m//n \to d_m$ rather than decreasing the hidden size of the encoder and decoder. This is in some ways unsurprising for a given step number as shown in the following figure as a larger-width model requires more compute to train, but closer inspection shows that it is also true when one normalizes for this condition (compare the 100k step loss of the $d_m=1024$ model to the 200k step loss of the $d_m=512$ model). In the following figure, we increase the number of layers to $n_l=16$ for both encoder and decoder.

![compression benefits](/deep-learning/compressed_vs_noncompressed.png)

As an aside, using $d_m=512$ transformers for both encoder and decoder for this dataset leads to a plateau a loss of around 5.0 by the equivalent of 50k steps, which is far worse than either mixer.

### Embedding-augmented causal language models

Recalling our original motivation to introduce input embeddings to remove the source entropy of text for a language model compressor, it may be wondered if one cannot combine the advantages of the autoencoder with next token prediction-based compression. The reson why this might be beneficial is as follows: it is apparent that masked mixer autoencoders require a compressed $d_m$ that is too large (in bits per $n_{ctx}$ tokens) to improve upon the compression found using next token prediction models given the relatively small amount of compute we have been applying to train these models. 

The primary reason for this is that each token (which usually representes between 3 and 5 bytes of text) is greatly expanded by the embedding transformation, whereby each token becomes represented by vectors of at least 512 elements, or at least 1024 bytes in fp16. This is such a large expansion that even our subsequent many-to-one token compression abilities do not give greater total compression than causal language models.

Some reflection may convince us that there is a good reason for this: it may be conjectured that our autoencoder less efficiently trainable than a next-token-prediction model (for our compute) because it must generate the entire context window in one forward pass, whereas the causal language model generates one next token at a time using information from all previous tokens. 

With this in mind, it may be wondered if we can combine the advantages of causal modeling and autoencoding to make an encoder-decoder architecture that exhibits superior compression ability to either autoencoder or causal language model alone, give similar compute to what has been used previously.

There are a number of options for how one could introduce the encoder's embedding into the decoder. Three straightforward candidates are concatenation in the token dimension, concatenation in the embedding dimension, or a linear combination in the embedding dimension. For embedding concatenation, we decrease the size of the token embeddings for the decoder commensurately to maintain a constant decoder size while comparing to other methods. These are illustrated in the following figure for convience.

![memory decoder architectures](/deep-learning/memory_encoder_options.png)

When we test the use of these three methods on FineMath 4+ data using a masked mixer decoder as our causal language model, we find that in general they are similarly efficient to train with the embedding concatenation method obtaining a slightly lower loss than embeddings combination or token concatenation. 

![memory decoder performances](/deep-learning/decoder_options.png)

One may expect for a transformer decoder to exhibit more training efficiency if given a token concatenation relative to embedding concatenation or combination, and indeed this is what we find (this time applied to the FineWeb-edu dataset):

![memory decoder performances](/deep-learning/transformer_memory_fig.png)

It appears from the above results that transformers are relatively invariant to how exactly the encoder's embedding is introduced among these three methods, so for text compression purposes we will use them interchangeably. 

Can adding an encoder's embedding lead to increased compression? To answer this we first need to know how large our embeddings are (particularly how many bytes they require) and then we can convert this to a bits-per-byte value.  Suppose one trains an embedding-augmented causal model where the embedding is of dimension $n_p$, each parameter being stored using $b_p$ bits, for a context window of size $n_{ctx}$ and $L_b / L_t$ bytes of input text per token. Then we can calculate the bits per byte required to store this embedding (amortized over the input) as previously via

$$
\mathtt{BPB} = \frac{n_p * b_p}{n_{ctx} * (L_b / L_t)}
$$

Once this value is known, we can find the loss offset $\Bbb L_o$ that corresponds to this extra required information,

$$
\Bbb L_o = \frac{\mathtt{BPB} * \ln (2)}{(L_t / L_b)} 
$$

and add this offset to the causal language modeling negative log likelihood loss for next token prediction to find the total loss.

$$
\Bbb L = \Bbb L(O(a, \theta), y) + \Bbb L_o
$$

We call this the 'normalized loss' for brevity. For a relatively small embedding ($d_m=64$) assuming 4 bits per parameter, and with a context window of $n_{ctx}=1024$ we have the following normalized loss (all masked mixers in the following figure are flat):

![memory decoder performances](/deep-learning/memory_compression_fig.png)

There is some expected behavior here: the embedding-augmented models begin training with higher loss (the result of the offset required for storage of the 64 floats in the embedding vector) but then approach or pass the purely causal model's loss (or equivalently its bpb compression of the input). 

It is somewhat less expected that the masked mixer decoder appears to be able to learn to use the information present in the embedding more efficiently than the transformer decoder, a pattern that is particularly apparent later in training. It is currently unclear why this would be. 

From the figure above, we may wonder whether an extended training run would lead to the embedding-augmented masked mixer overtaking the transformer with respect to normalized log likelihood loss (ie total compression). We find that this is indeed the case: training on more samples (with the same lr scheduler and batch size etc.) leads to the masked mixer achieving the lowest total bits per byte, assuming 4 bits per parameter for the embedding.

![memory decoder performances](/deep-learning/extended_memory_figure.png)

From these results it appears that the informational content of the embedding (only 64 parameters in this case) is not saturated even after a relatively long training run such that the loss continues to decrease nearly linearly. It seems safe to assume that the embedding-augmented mixer would be a more efficient compression model for a fixed compute applied at training even if the embedding were quantized with 8 or 16 bits per parameter.

The above results are obtained for flat masked mixers, and as we observed superior training efficiencies with multi-headed mixers of the same size one may expect to find better training properties for embedding-augmented causal mixers as well...

### Memory Models

The ability to compress information from a sequence of tokens into one embedding in an efficient manner has another utility: we can use these embeddings to provide exended context to a model without increasing its inference computation.

The architecture we will experiment with here is mostly similar to the embedding-augmented causal langauge model architecture implemented above, where we use the token dimension concatenation to maximize the number of embeddings we can provide to the decoder model.

![memory decoder architectures](/deep-learning/memory_transformer.png)

The notable difference between this architecture and the token concatenation-based autoencoder introduced above is that we not longer care about compressing the embedding fed from the encoder to decoder. This is because if one uses token concatenation, each token in the decoder is converted to a vector of dimension $d_m$ such that it is natural to supply the encoder's embedding as a vector of that same dimension. This also allows us to provide embeddings of $n$ encoded text sequences as $n$ embeddings, taking the place of $n$ tokens of the decoder. It is clear to see that this is much more efficient in terms of decoder input space than embeding concatenation.

The idea of compressing input sequences into embeddings that take the place of transformer token embeddings is not new, and was explored in various forms (see particularly the [recurrent memory transformer](https://arxiv.org/abs/2207.06881)). Such models were shown to be able to perform simple information retrieval (needle-in-haystack style) on the compressed inputs but little more, and certainly do not retain most information of the input. 

The insight here is that as we have seen that transformers are quite inefficient with respect to capturing input information in encodings, we can use masked mixer encoders instead to greatly increase the encoder's fidelity.



 

