## Arbitrary Retrieval

### Background

Suppose we want to use pretrained language models for the purpose of retrieving some target passage given a query. In the general case this is a form of search: given a search query, what passage contains information on that query?

A few years ago, language retrieval 

### Lossless Embeddings via high-fidelity compressions

### Training for arbitrary retrieval 

### InfoNCE retrieval training leads to reduced information retention

### Why InfoNCE reduces information

It may be wondered whether the source of this decrease in embedding information upon retrieval training is the training setting or whether InfoNCE minimization itself is to blame. As we have made some effort to ensure that the retrieval setting requires lossless compression, we can guess that it is the objective itself that causes this change. And a little theory is enough to convince one that this is indeed the case.

### Retrieval without Information Loss

Perhaps the most straightforward way to prevent information loss is to dispense with model-free retrieval and use a trainable model to match query and target embeddings. To see why this is the case, consider a very general model-free retrieval training setting: one has a set of query and target embeddings and wants to use some operation to measure the similarity between pairs of queries and targets. Minimization of the cosine distance between these pairs is commonly accomplished via InfoNCE, but however it is done the resulting vectors of query and target are ideally indistinguishable by some metric (cosine distance, normed distance, or another metric). The problem is that even if we do not minimize InfoNCE, this setting itself implies a symmetric maximization as Banach (generalized vector) space metrics are themselves symmetric.

Replacing metric minimization with a trainable model allows us to bypass this difficulty of symmetric information entirely. The simplest way to maintain a high-fidelity encoder's input information during the retrieval training process is to simply freeze the encoder and train only the retrieval model, where the retrieval model is fed a sequence of $n_{ctx}$ embeddings of dimension $d_m$ and maps these to a probability distribution over all target embeddings (for single-query retrieval, $n_{ctx}-1$ of them). This approach was introduced [here](https://arxiv.org/abs/2409.01482), although for summary/story query/target matching was found to be less efficient that standard InfoNCE (which is unsurprising given the findings on this page). 

### Image-Language Modeling

The use of InfoNCE to align embeddings extends beyond retrieval, with this approach forming the basis for most image-language modeling as it is the method CLIP and similar objectives match text and images. But this presents a significant problem: assuming that one has trained an image-language model via CLIP (minimizing InfoNCE) and can generate images by feeding a model text (or vice versa), in the model's view the text you give the model and the resulting image are identical in information content. 

