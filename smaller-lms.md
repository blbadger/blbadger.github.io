## Smaller, Simpler Langauge Models

### Background 

The training of the most effective language models today (3/2024) requires an enormous amount of computational resources: a whopping 1720320 hours of 80GB nvidia A100 compute time were required to train the 70 billion parameter version of [Llama 2](https://arxiv.org/pdf/2307.09288.pdf). Assuming that the meta RSC was used (6080 GPUs), this comes out to nearly two weeks of nonstop training for that entire cluster.  The cost to reproduce this training on the cloud at the same day would be around four million usd., making this kind of training prohibitively expensive for all except a few large organizations.

This prohibitive amount of compute power required is mostly down to the very large size of the models that are currently trained: most LLMs are trained on something like 2 trillion tokens of text, but this is not actually that much information when one considers that each token is typically in bytecode and therefore the training dataset is around 2 TB, substantially smaller than the large image datasets required to train (much smaller) diffusion models.

To understand why this is the current state of affairs, it is enlightening to remember firstly that all current large language models are based on the transformer architecture, and that that architecture was origially introduced with the motivation of looking for a model that would not 'saturate', is it gets better the more tokens one uses to train it and the larger the model is. It is therefore not very surprising that the current large language models are very large indeed, as the architecture they are built on was originally found to benefit from an enormous number of parameters.

This begs the question: are these parameters necessary? It is clear that transformer-based models do indeed become substantially more effective with an increase in the number of parameters they contain, but if we were not restricted to one particular architecture is it possible that we could design a model with far fewer parameters for language modeling?

A very quick calculation suggests that billions or even millions of parameters are far more than would be necessary to model the English language. It has been claimed that there are somewhere around $10^570$ possible English sentences, as an upper bound. Without knowing how to model these sentences, we can view them as $10^570$ points in an arbitrarily high-dimension space. Now due to a theorem of high-dimensional space, the same points may be obtained with arbitrary precision in a space that is approximately $\log 10^570 = 570$ dimensional.  This means that a model with the same number of parameters may exist such that each sentence may be found via a combination of these parameters.

Although it has been found that for restricted inputs even smaller models (millions rather than billions of parameters) may give coherent outputs.

[Elsewhere](https://blbadger.github.io/language-discreteness.html) it was observed that 
