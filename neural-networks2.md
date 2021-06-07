## Neural Networks II: Interesting features

This page is a direct continuation from from [part one](https://blbadger.github.io/neural-networks.html). Code for this page is found in [this Github repo](https://github.com/blbadger/nnetworks).

### Adversarial training datasets

From the theoretical considerations presented [here](https://blbadger.github.io/nn-limitations.html) and elsewhere, it is clear that one inherent limitation in neural nets is the propensity for nearly indistinguishable (in the feature space) inputs to be mapped to very different outputs.  Input examples such as these are called 'adversarial examples'. 

The collection of images used to train a network may be thought of as a single input (broken up into smaller pieces for convenience).  The theorem referenced above is not particular to single input images but instead applies to any input, leading to the following question: could nearly indistinguishable training datasets yield very different outputs?  Here outputs may be the cost (or objective) function values summed accross all inputs, or (more or less equivalently) the average classification accuracy.

Say we have three datasets of similar size and content.  These correspond to slightly modified versions of those used in the [last section]([here](https://blbadger.github.io/nn-limitations.html), and are available in the github repository linked on this page above.

First there is the original training dataset,
![training](/neural_networks/train.png)

then there is the first test dataset,
![test 1](/neural_networks/test1.png)

and a second test set
![test 2](/neural_networks/test_2.png)

These three datasets are similar in appearance, and are actually images of cells from three different fruit fly brains.  There is a clear pattern between the labels 'Snap29' and 'Control': there are numerous blobs inside the cells of the former but not the latter.  

Now let's train the deep convolutional network on each dataset, using the other two datasets as the test sets for observation of training efficacy.

![training results](/neural_networks/nn_training.png)

