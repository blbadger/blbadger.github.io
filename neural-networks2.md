## Neural Networks II: Interesting features

This page is a direct continuation from from [part one](https://blbadger.github.io/neural-networks.html). Code for this page is found [here](https://github.com/blbadger/nnetworks).

### Adversarial training datasets

From the theoretical considerations presented [here](https://blbadger.github.io/nn-limitations.html) and elsewhere, it is clear that one inherent limitation in neural nets is the propensity for nearly indistinguishable (in the feature space) inputs to be mapped to very different outputs.  Input examples such as these are called 'adversarial examples'. 

The collection of images used to train a network may be thought of as a single input (broken up into smaller pieces for convenience).  The theorem referenced above is not particular to single input images but instead applies to any input, leading to the following question: could nearly indistinguishable training datasets yield very different outputs?  Here outputs may be the cost (or objective) function values summed accross all 

Say we have three datasets of similar size and content.

![training](/neural_networks/)

![test 1]()

![test 2]()

Now let's train the deep convolutional network on each dataset, using the other two datasets as the test sets for observation of training efficacy.

![training results](/neural_networks/nn_training.png)

