## Language Model Representations and Features

### Introduction to Language

One of the most important theorems of machine learning was introduced by [Wolpert](https://direct.mit.edu/neco/article-abstract/8/7/1341/6016/The-Lack-of-A-Priori-Distinctions-Between-Learning) and is colloquially known as the 'No Free Lunch' theorem, which may be stated as the following: no particular machine learning method is better or worse than any other when applied to the problem of modeling all possible statistical distributions.  A similar theorem was shown for the problem of optimization (which covers the processes by which most machine learning algorithms learn) by [Wolpert and Macready](https://ieeexplore.ieee.org/abstract/document/585893), and states that no one algorithm is better or worse than any other across all classes of optimization problems.

These theorems are often used to explain overfitting, in which a model may explain training data arbitrarily well but explains other similar data (usually called validation data) poorly.  But it should be noted that most accurately these theorems do not apply to the typical case in which a finite distribution is being modeled, an d therefore are not applicable to cases of empirical overfitting.  But they do convey the useful idea that each machine learning model and optimization procedure is usueful for only some proper subset of all possible statistical distributions that could exist.

The assumption that not all possible distributions may be modeled by a given learning procedure does not affect the performance of machine learning, if one assumes what is termed the manifold hypothesis.  This hypothesis posits that the set of all elements for a given learning task (ie natural images for a vision model or English sentences for a language model) is a small subset of the set of all elements that could exist in the given space (ie all possible images of a certain resolution or all possible permutations of words and spaces).  The task of the learning program is simply to find the manifold, which is the smooth and approximately connected lower-dimensional space that exists in the higher-dimensional input, such that the objective function associated with the learning task is minimized.

It has been observed on [this page](https://blbadger.github.io/depth-generality.html) that subsequent layers of vision models tend to map arbitrary inputs to manifolds learned during training, rather than simply selecting for certain pieces of input information that may be important for the task at hand. The important difference is that we can see that vision models tend to 'infer' rather than simply select information about their input during the manifold mapping process.  Typically the deeper the layer of the vision model, the more input information is lost in the untrained model and the more information that is inferred in the trained one.

We may wonder whether this is also the case for language models: to these also tend to infer information about their input? Are deeper layers of language models less capable of accurately reconstructing the input, as is the case for vision models?  Do deeper layers infer more input information after training?

### Spatial Learning in Language Models

The most prominent deep learning model architecture used to model language is the Transformer, which combines dot product self-attention with feedforward layers applied identically to all elements in an input to yield an output.  Self-attention layers originally were applied to recurrent neural networks in order to prevent the loss of information in earlier words in a sentence, a problem quite different to the ones typically faced by vision models. 

Thus convolutional vision and language models diverged until it was observed that the transformer architecture (with some small modifications) was also effective in the task of image recognition.  Somewhat surprisingly, [esewhere](https://blbadger.github.io/transformer-features.html) we have seen that transformers designed for vision tasks tend to learn in a somewhat analagous fashion to convolutional models: each neuron in the attention module's MLP output acts similarly to a convolutional kernal, in that the activation of this neuron yields similar feature maps to the activation of all elements in one convolutional filter.

It may be wondered whether transformers designed for language tasks exhibit some of these same features.

![gpt2 feature visualization]({{https://blbadger.github.io}}/deep-learning/gpt2_features_viz.png)


![gpt2 feature visualization]({{https://blbadger.github.io}}/deep-learning/gpt2_features_viz_2.png)


![gpt2 feature visualization]({{https://blbadger.github.io}}/deep-learning/gpt2_features_viz_3.png)

### Input Representation with Language Models



### Language models translate gibberish into words

