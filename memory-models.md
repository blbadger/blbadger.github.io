## Language Mixers IV: Memory Models

This page focuses on information compression, specifically how we can achieve better compression via new language model architectures and how this can be used for large-context models.

### Introduction

In work detailed in [this blog post](https://blbadger.github.io/smaller-lms-2.html) and [this paper](https://arxiv.org/pdf/2409.01482), a major finding was that one can modify a new causal language model architecture termed the 'masked mixer' (which is essentially a transformer with self-attention replaced by masked 1D convolutions) to effectively autoencode inputs with high compression, that is, one can train small models in reasonable time to be able to regenerate (with some error) a sequence of 512 or more tokens using the embedding of only one token with excellent generalization properties. It was found that using masked mixers for encoder and decoder allows for far greater input autoencoding accuracy than a transformer encoder/decoder pair for a given model size ad compute budget during training, leading to the ability to compress inputs using new and potentially more efficient methods than what has been possible using the transformer.

It may be wondered why text compression ability is important: even if large language models achieve 2x or 4x the compression of commonly used algorithms like `gzip`, they are thousands of times more computationally expensive to actually run and thus not preferred today (although this may change in the future). The answer is that effective compression from many to one token allows one to design architectures that have interesting properties such as extended context windows or meta-context guidance, which we will investigate under the broad name of 'memory'.

We will first tweak the autoencoder mixer architecture to try to obtain optimal text compression in fixed compute budgets before using this information to attempt to test whether one can obtain better text compression ratios than the current best methods (ie large transformers). We will conclude by examining the suitability of autoencoding embeddings for extending generative language model context with sub-quadratic complexity using encoder embeddings.

### Transformer-based autoencoders train and scale poorly

In [previous work](https://arxiv.org/pdf/2409.01482) evidence was presented for the idea that masked mixers were far better autoencoders than transformers, with the primary large-scale data evidence being the following: if one trains a $d_m=1024$ masked mixer autoencoder with $n_l=8$ layers in the encoder and decoder, one reaches a far lower CEL than the compute- and memory- transformer model with $d_m=512$ (transformers contain far more activations per $d_m$ due to their $K, Q, V$ projections). One may object that this could be mostly due to differences in the amount of information available in a 512-dimensional vector compared to a 1024-dimensional vector, although that was not found to be the case on the much smaller TinyStories dataset where equivalent-dimensional transformers far underperformed their masked mixer counterparts despite requiring around double the compute and device memory.

We are in a position to now further explore the training efficiencies for transformer-based versus mixer-based autoencoders. Perhaps the first question one might have when observing the very large performance gap between transformer and mixer is if there is some implementation error in the transformer architecture. The original implementation used pre-built Llama model blocks from the Huggingface transformers library, and implemented loops through these blocks for encoder and decoder while using the same vector broadcasting and tensor rearrangements, word-token embedding, and language modeling heads transformations as the masked mixer counterpart, feeding the positional information to each layer. It may be wondered if a much simpler implementation would perform better, where the encoder and decoder are each `LlamaModel` transformer implementations, and we also include an attention mask on the decoder. From the figure below, we can see that the transformer autoencoder with pre-built encoder and decoder is actually somewhat worse than the original modular architecture when trained on the FineMath 4+ dataset.

![prebuilt vs module transformer autoencoder](/deep-learning/prebuilt_transformer_autoencoder.png)

The other main oustanding question is whether the increased masked mixer autoencoder training efficiency might be due to the larger embedding dimension in that model versus the transformer, at least for larger and more diverse datasets like finemath 4+ or fineweb-edu. This is primarily a scaling question with respect to increases in the model $d_m$ (and thus the embedding dimension in this case), so one can obtain a more general idea of the differences in masked mixer versus transformers for autoencoder training efficiency by comparing the losses achieved as one scales the model $d_m$ for a given training dataset.

From the following figure, it can be appreciated that indeed transformers are far less efficient to train as autoencoders than masked mixers for multiple $d_m$ values, providing evidence for the idea that differences in autoencoding abilities between these models are not due to those differences but are instead model intrinsic (specifically attention-intrinsic). For context, we have a total of $n = 200,000 * 128 * 512 \approx 13.1 * 10^9$ tokens trained at 200k steps for each model in the following figure.

![transformer versus mixer autoencoders](/deep-learning/autoencoder_scaling_figure.png)

It is apparent that transformers scale badly in terms of samples per model, as apparent by the negative asymptotic slope of the transformer training curves being far smaller than that of the masked mixer. Transformers-based autoencoders also scale poorly in terms of scaling the embedding size or equivalently the model width, which is apparent as doubling the $d_m$ of a transformer autoencoder twice gives decreasing loss achieved with each doubling, whereas the opposite is true for masked mixer autoencoders.

None of this is particularly surprising given the results and theory outlined in the masked mixer [introductory paper](https://arxiv.org/pdf/2409.01482). One major finding there is that transformers are relatively inefficient to train for tasks requiring retention of information present in the input, either in a single or multiple output embeddings. 

### Causal masking increases autoencoder training efficiency

To begin with, it is helpful to recall the architecture of the masked mixer-based autoencoder as presented in the work linked in the last section:

![mixer autoencoder architecture](/deep-learning/mixer_autoencoder.png)

This architecture recieved little to no optimization in that work, as it was mostly presented as evidence for a hypothesis involving bijective functions and autoencoding. But now that we want to improve and elaborate upon this architecture, where would we start? 

One obvious question is whether or not the convolutions really need to be masked: after all, we are generating the output in one step and are not adhering to the causal language modeling objective of next token prediction, so why would we really want to mask the model's convolutions? Bearing this in mind, it is somewhat unexpected that removing the encoder convolution's mask results in substantially less efficient training and even more surprising that removal of the decoder's mask (keeping the encoder unmasked as well) results in highly unstable training with gradient norm spikes leading to rapid rises in loss during training. The following figure details the cross-entropy losses achieved by masked mixer-based autoencoders ($d_m=1024, n_{ctx}=512, n_l=8, b=128$) on the `FineWeb-edu (10BT)` dataset, where the evaluation is a hold-out sample from the corpora (which has been measured to contain <5% duplicate documents). Here 'masked' indicates a causal (left-to-right) mask implemented using a lower triangular mask on the 1D convolution weight matrix, and the right panel is simply an extended training run of the conditions in the left panel (omitting the unstable no-masked model).

![mixer autoencoder efficiencies](/deep-learning/autoencoder_causality.png)

Why would causal masking be so important to a model that does not perform causal modeling? There is usually some benefit to matching a model's structure to any prior we know about the dataset that is being modeled, and with that perspective one could perhaps guess that enforcing causality is beneficial because the data being modeled (text) is in some way fundamentally causal as it is understood in one orientation. It is less certain why removing all causality masks leads to highly unstable training, as one may expect for a simple decrease in efficiency in this paradigm rather than exploding gradients. 

### Why transformers struggle in this autoencoding paradigm

From the last section we observed that transformers make far worse autoencoders (with the architecture presented at least) than masked mixers. The next question to ask is why this is: why would transformers be so much worse than masked mixers, given that in previous work has shown more modest differences in information retention between transformers and masked mixers? More specifically, why would attention lead to poor autoencoding in this architecture?

Some reflection will provide a reasonable hypothesis: recall that attention may be expressed as follows:

$$
A(Q, K, V) = \mathrm{softmax} \left( \frac{QK^T}{\sqrt(d_k)} \right)V
$$

where the $Q, K, V$ are matrices of packed vectors projected from each token embedding. To make these operations clearer from the perspective of token indices, we can ignore the $d_k$ scaling factor and  express this equation as follows: the attention between the query projection for token $i$ and key and value projections at index $j$, for ${i, j \in n}$ is

$$
A(q_i, k_j, v_j) = \frac{\exp \left( (q_i \cdot k_j) v_j \right)}{ \sum_n \exp \left( (q_i \cdot k_n) v_n \right)}
$$

Now consider what we have done by repeating the input embedding for all tokens in the decoder: as the projection weight matrices $W_k, W_q, W_v$ are identical for all tokens, the necessarily $k_i = k_j \forall i, j$ and similarly $q_i = q_j \forall i, j$ and $v_i = v_j \forall i, j$ and thus $q_i \cdot k_j = q_i \cdot k_l \forall i, l$. Therefore $A(q_i, k, v) = A(q_j, k, v) \forall i, j$ such that our outputs from attention are identical across all token embeddings.

Given this observation, it is not hard to see that this identicality will persist for more than one layer as each feedforward module following attention is identical. Thus it is little wonder why transformers would perform so poorly when applied to repeated input embeddings.

Is there experimental evidence for this idea? We can test the performance of various autoencoder topologies to look for some.

Given some evidence for our idea, how would one go about injecting an encoder's embedding to a transformer decoder while avoiding the identical attention problem? One simple but effective way to do this is to 'unroll' the embedding by projecting unique subsets of the encoder's embedding into each of the decoder's input positions. A relatively simple but flexible method to get unique subsets of a vector is to use a sliding window, where we project from a certain number of contiguous elements of a vector in our 'window' and shift the elements projected from at each of the decoder's indices, keeping the projection weights identical across all input windows. This requires an embedding vector satisfying $d_m \geq n_{ctx}$ for every decoder index vector to be unique, but we can simply add a projection to enlarge the $d_m$ of the embedding as required if necessary. 

For cases where we want to project from $n_d$ elements and $d_m < n_d + n_{ctx} - 1$, or in other words where our window slides off our embedding vector to make all decoder inputs, we can simply wrap our window around to the first index of the embedding, concatenate, and project accordingly. For the case where $n_ctx=4, d_m=6, n_d = 4$, the following diagram illustrates the projection and wrapping process:

![mixer autoencoder efficiencies](/deep-learning/sliding_window_embedding.png)


This can be implemented as follows: given a linear projection layer that assumes the input is half the size of the ouput, `self.projection = nn.Linear(dim//2, dim)`, we can replace the embedding repeat with
our unrolled projections as follows:

```python
encoder_embedding = x[:, -1, :].unsqueeze(1) # dim=[batch, token, hidden]
embedding_stack = []
# sliding window unroll over hidden dim
for i in range(self.tokenized_length):
    sliding_window = encoder_embedding[..., i:i+dim//2]
    if i+dim//2 > dim:
        residual = i+dim//2 - self.tokenized_length
        # loop around to first index
        sliding_window = torch.cat((sliding_window, encoder_embedding[..., :residual]), dim=2)
    embedding_stack.append(sliding_window)
encoder_embedding = torch.cat(embedding_stack, dim=1)
encoder_embedding = self.projection(encoder_embedding)
```

Note here that an implementation most faithful to our figure above would be to apply the projection at each index in the for loop before concatenation, but this is much less efficient as applying the projection to the pre-concatenated output allows us to make use of device (GPU) parallelization that is otherwise tricky to add to the loop via Pytorch primitives.

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

From the figure above one clearly reaches limited returns when expanding beyond four heads as there is a significant computational burden but very little efficiency benefit with an increase to 8 heads from 4, and an efficiency detriment if we increase to 16 heads from 8. It is curious that four heads are also optimal for causal transformer models of similar size with respect to loss achieved per unit of compute applied during training for this dataset as well.

Interestingly, however, the multi-headed mixer autoencoder experiences instabilities late in training manifesting in very rapidly exploding gradients for models with one or two heads. As this was observed early in training for autoencoders without causal masks, one straightforward explanation would be a problem with the multi-headed mixer's masks. We causally mask the convolution in each head, and a quick test shows that the encoder and decoder modules are indeed causal. A relatively straighforward solution for this problem would be to add a layer or RMS normalization to the concatenated projections, or add residuals across head elements. We can avoid these architectural changes by carefully choosing a datataype, however, as explained below.

A straightforward way to address gradient overflow is to use a datatype with a wider dynamic range than fp16 (which is by definition e5m11), for example either fp32 (e8m23) or the nearly equivalently expressive bf16 (e8m7). bf16 is supported on newer hardware (A100s and H100s and RTX 30 series and above, and although it can be emulated with V100s this emulation cannot make use of the tensor cores in that GPU and is thus low-throughput) and is well supported for bf16/fp32 mixed precision training integration by Pytorch. When we train the same models using bf16/fp32 precision, we no longer observe catastrophic instabilities in multi-headed autoencoder trainings (see the above figure for an example) which supports the hypothesis that numerical overflow is the cause of training instabilities in multi-headed mixer autoencoders.

There is a substantial decrease in loss per step for causal autoencoders trained on the FineWeb as well, although we also find exploding gradients leading to a catastrophic increase in training loss for multi-headed mixer autoencoders. As the exploding gradient and loss problem appears pervasive for multi-headed masked mixer autoencoders, one can attempt to understand why this is the case. A good first candidate could be the datatype we are using for training which is fp16/fp32 mixed precision to allow for older hardware (V100) compatibility. Although this training is usually decently stable across a wide range of model, one can argue that an autoencoder of this type is inherently unstable with respect to gradient norms as all gradients flow through one vector, which is susceptible to numerical overlfow if the vector's partial derivatives are sufficiently large. 

![decoder options](/deep-learning/fineweb_autoencoder_heads.png)

One can increase the number of trainable inter-token parameters in a simpler way than using multiple heads: using a convolutional kernel of size greater than unity (k=2 here denotes two kernels) scales the number of inter-token parameters as 

$$
p=n_l (k * n_{ctx}^2)
$$

as there are simply $k$ weight maps per convolutional layer. From the figure it is apparent that a flat masked mixer with k=8 achieves identical loss to the 4-headed mixer, but does not suffer the numerical instabilities associated with the latter.

It is also noteworthy that there is such a large difference in training efficiencies for multi-headed versus flat masked mixers for autoencoders. One can estimate the increase in training efficiency by finding the number of samples required to reach a certain loss value, and in this case one requires more than 6x the number of samples for the flat masked mixer to approach the loss achieved by the 4- or 8-headed autoencoder at 200k steps. For comparison, the difference in causal language model loss per step between flat and multi-headed mixers is very small: the flat mixer requires only around 1.05x the number of samples to match the 4-headed mixer when trained on TinyStories. If we implement a causal masked mixer while keeping the projection dimension equal to $d_m/n_h$, we find that there is very little difference in per-step loss between this model and the flat masked mixer when trained on the FineWeb-edu (10BT) dataset.

![causal mixer heads](/deep-learning/causal_mixer_fineweb_heads.png)


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

### Embedding-augmented causal language models

Recalling our original motivation to introduce input embeddings to remove the source entropy of text for a language model compressor, it may be wondered if one cannot combine the advantages of the autoencoder with next token prediction-based compression. The reson why this might be beneficial is as follows: it is apparent that masked mixer autoencoders require a compressed $d_m$ that is too large (in bits per $n_{ctx}$ tokens) to improve upon the compression found using next token prediction models given the relatively small amount of compute we have been applying to train these models. 

The primary reason for this is that each token (which usually represents between 3 and 5 bytes of text) is greatly expanded by the embedding transformation, whereby each token becomes represented by vectors of at least 512 elements, or at least 1024 bytes in fp16. This is such a large expansion that even our subsequent many-to-one token compression abilities do not give greater total compression than causal language models.

Some reflection may convince us that there is a good reason for this: it may be conjectured that our autoencoder is less efficiently trainable than a next-token-prediction model (for our compute) because it must generate the entire context window in one forward pass, whereas the causal language model generates one next token at a time using information from all previous tokens. 

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

The above results are obtained for flat masked mixers, and as we observed superior training efficiencies with multi-headed mixers of the same size one may expect to find better training properties for embedding-augmented causal mixers as well. It comes a some suprise, therefore, that an embedding-augmented causal masked mixer (using embedding dimension concatentation) with four-headed mixer layers in the encoder slightly underperforms the same model without heads as shown in the figure below.

![memory decoder performances](/deep-learning/ememory_encoder_heads.png)

A similar result is obtained where we instead increase the number of inter-token parameters by using convolutional kernels of size four rather than one, where we see no increase in training efficiency upon doing so when the encoder's embedding is introduced via embedding concatenation but curiously not when we use token dimension introduction for the embedding, in which case there is a notable improvement in training efficiency.

![memory decoder performances](/deep-learning/token_memory_figure.png)

### Memory Models

The ability to compress information from a sequence of tokens into one embedding in an efficient manner has another utility: we can use these embeddings to provide exended context to a model without increasing its inference computation. Extensive research has been performed on methods to reduce the amount of computation and memory required to train and perform inference on a model applied to $n$ tokens, and this problem has been particularly relevant to recent advances in code generation, mathematical problem solving, and other domains benefitting from chain-of-thought test-time compute scaling. In this paradigm, the performance of a model scales with the number of tokens one generates (before generating a final answer) such that inference compute and memory become significant bottlenecks. 

On the compute side, transformers scale with $O(n^2)$ for $n$ tokens (assuming $K, V$ caching) making very long context window inference prohibitevly expensive unless one has access to large GPU clusters. This is true regardless of whether one uses a mixture of experts or attention innovations such as multi-headed latent attention, as these only introduce constant factors to the scaling complexity. Memory requirements are $O(n)$ during inference if $K, V$ caching is used and $O(n^2)$ during training as caching cannot be used as one must backpropegate across all activations. 

Decreasing both memory and compute growth is an active area of current research, with most efforts aligned with attempts to use sub-quadratic complexity attention or attention alternatives. Here we take a different approach: our generative model is still $O(n^2)$ compute (and $O(n)$ memory if caching is implemented) for inference regardless of mixer or transformer use, but we provide embeddings representing entire sequences as inputs to that model in certain indices rather than only embeddings representing tokens. For the case where we have one encoder model and one decoder of a fixed length, the compute required is $O(n)$ with length (a slight abuse of notation as this is only valid up to the length $n_{ctx}^2$) as one simply uses more of the decoder's embedding indices for sequences rather than tokens, and similarly the memory scaling is $O(1)$ again only up to $n_{ctx}^2$.

The idea of compressing input sequences into embeddings that take the place of transformer token embeddings is not new, and was explored in various forms (see particularly the [recurrent memory transformer](https://arxiv.org/abs/2207.06881)). Such models were shown to be able to perform simple information retrieval (needle-in-haystack style) on the compressed inputs but little more than that, and certainly do not retain most information of the input. The insight here is that as we have seen that transformers are quite inefficient with respect to capturing input information in encodings but masked mixers are not, we can use masked mixer encoders instead to greatly increase the amount of information that is stored in each embedding.

The architecture we will experiment with here is mostly similar to the embedding-augmented causal langauge model architecture implemented above, where we use the token dimension concatenation to maximize the number of embeddings we can provide to the decoder model.

![memory decoder architectures](/deep-learning/memory_transformer.png)

The notable difference between this architecture and the token concatenation-based autoencoder introduced above is that we not longer care about compressing the embedding fed from the encoder to decoder. This is because if one uses token concatenation, each token in the decoder is converted to a vector of dimension $d_m$ such that it is natural to supply the encoder's embedding as a vector of that same dimension. This also allows us to provide embeddings of $n$ encoded text sequences as $n$ embeddings, taking the place of $n$ tokens of the decoder. It is clear to see that this is much more efficient in terms of decoder input space than embeding concatenation, and avoid sthe problems of input ambuguity present when performing linear combinations in the embedding dimension.

The first question we can ask is as follows: can a causal decoder make use of the information present in an autoencoder's embedding? Essentially the question we want to address here is whether the information present in the embedding passed from encoder to decoder in the autoencoding architecture (where all tokens are generated in one forward pass) is beneficial for the case where we train a causal decoder (where each forward pass yields only one token). We can test this by observing training efficiencies for a causal masked mixer and an embedding-augmented masked mixer, where the latter may be implemented by training an autoencoder, discarding the decoder, adding a causal decoder, and freezing the encoder's parameters during training. To save memory and compute during training, one could simply save the embeddings generated by this autoencoder to storage and reference them when training and evaluating the decoder, but we choose the former implementation to avoid a significant storage overhaed.

![memory decoder architectures](/deep-learning/frozen_memory_figure.png)

Now that we have seen that casual decoders can indeed make use of encoded information, we can investigate the question of efficiency: is it better to train both encoder and causal decoder simultanously, or use an encoder from a previously-trained autoencoder and train the causal decoder separately? As shown in the left figure below, at least for a non-optimized encoder the answer is that training both simultaneously is far more efficient. A significant caveat here is that the encoder used in this experiments is fairly weak: the autoencoder it came from only obtained a CEL of $\Bbb L \approx 3.3$ on this dataset, which is not particularly low.

![memory decoder architectures](/deep-learning/fineweb_memory_figure.png)
 
As is the case for memory models designed for information compression (ie with very small embeddings), a multi-headed mixer showed no benefit over the flat masked mixer early in its training run. Instead, we see a precipitous drop in cross-entropy loss when increasing the convolutional kernel to four (from one) to the point that our $d_m=1024$ memory model is able to effectively perform a $512 \to 1$ compression on token embeddings after training for half a day on two H100s with very little error resulting from this compression. 

How much of the difference in training loss between frozen and standard memory models (shown in the above left) is due to the relatively high loss achieved by the autoencoder? As training encoder and decoder separately has significant advantages with respect to memory and compute required, so it is worth looking into what is required for effective frozen memory model training as well.

![memory decoder architectures](/deep-learning/fineweb_memory_d1024_fig.png)


