## Masked Mixers II

We continue exploring topics from [Part I](https://blbadger.github.io/smaller-lms.html), with a more thorough version of these results found in [this paper](https://arxiv.org/pdf/2409.01482)
Application of masked mixers to larger datasets will be explored, theory is expanded, and the nature of the learned structure of masked mixers versus transformers is investigated.

### Why Masked Mixers?

In [Part I](https://blbadger.github.io/smaller-lms.html) the motivation behind masked mixers for language modeling was detailed, but can be restated briefly: from high-dimensional projection theory (specifically the Johnson-Lindenstrauss lemma) we can accurately represent points in high-dimensional space in a space with much fewer dimensions (approximately the log of the number of points we have). This is an important result here because it implies that one can make language models that are capable of fitting extremely large datasets with what is currently considered an extremly small number of parameters: for example, a dataset one million times the size of the already large 15 trillion token Llama-3.1 dataset ($1.5 * 10^19$ tokens to be precise) may be arbitrarily well fitted by a 352-dimensional model, which would require around 100 million parameters for a llama-style generative model.

Empirically it appears that the current state-of-the-art model type (the transformer) trains much too slowly to achieve this feat in any reasonable length of time, and the best-performing models are typically trained on thousands of GPUs for many days. From other investigations on information transfer between model layers, we wondered whether an architecture that more accurately represents its inputs (the masked mixer) would learn more efficiently.

Restricting our investigations to a small dataset (TinyStories, ie small stories written by ChatGPT in the style of a four-year-old) we found that although masked mixers are much more efficient learners than the original GPT-style transformers, highly optimized and current transformers learn slightly more efficiently. Much development has occurred to make the transformer as efficient as it is, so one of the goals for this page is to improve the masked mixer to exceed current transformer efficiency. Masked mixer was found to be much more efficient for language retrieval, a task that will be considered further here as well.

### Accuracy and Flexibility

Even with these promising features, the question may be asked: why masked mixers? Before proceeding to further investigations on this architecture, it is worth considering what you get when you swap self-attention for masked convolutions.

Before masked mixers were first trained, it was found that these models give much more accurate representations of their inputs than transformers. In effect, this means that given an input, the information necessary to uniquely identify that input via gradient descent is retained throughout the mixer but not the transformer. It could be argued for or against the idea that this would be useful for language generation, and perhaps more convincingly argued that this is important for langauge retrieval. But accurate input representation remains a useful feature of mixers for a variety of applications. We will refer to this input representation *accuracy* more in the future.

Perhaps as important is the model architecture *flexibility* of masked mixers. From investigations into representation in [vision transformers](https://blbadger.github.io/vision-transformers.html), it was apparent that vision transformers require each key component of their architectures: layer normalizations, MLPs, self-attention, and residual connections are all required for effective gradient propegation.

This can be tested more directly by simply removing each component and observing the training process. Ironically for an architecture introduced as 'Attention is all you need', self-attention is actually the only removable component (as long as it is replaced by some other trainiable inter-token parameters): removal of MLPs (or replacing with attention) layer norms or residual connections results in very poor language model training: either failure to minimize a loss function (if MLPs are removed or replaced with attention) or training becomes unstable (for removal of layer norms or residuals) and loss spikes to infinity. The reason for this is that attention is a rather difficult transformation for gradients to propegate across, and this is important because it essentially 'fixes' the architectures of models with attention to similar patterns, all requiring some form or other of the same components transformers have (layer norms, residuals, MLPs, ets.).

On the other hand it turns out that langauge training proceeds perfectly well (although slightly less efficiently) when layer norms or residuals are removed from the masked mixer architecture. This means that the mixer architecture is effectively much more flexible than the transformer, and can be modified to a much greater extent. This topic will be explored more in the 'Linear mixer' section of this page.


### Language Generation Training Efficiency




### Linear Mixers






