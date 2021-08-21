## Neural Networks II

This page is a direct continuation from from [part one](https://blbadger.github.io/neural-networks.html). Code for this page is found in [this Github repo](https://github.com/blbadger/nnetworks).

-----------------Currently under construction-------------

### Adversarial training datasets

From the theoretical considerations presented [here](https://blbadger.github.io/nn-limitations.html) and elsewhere, it is clear that one inherent limitation in neural nets is the propensity for nearly indistinguishable (in the feature space) inputs to be mapped to very different outputs.  Input examples such as these are called 'adversarial examples'. 

The collection of images used to train a network may be thought of as a single input (broken up into smaller pieces for convenience).  The theorem referenced above is not particular to single input images but instead applies to any input, leading to the following question: could nearly indistinguishable training datasets yield very different outputs?  Here outputs may be the cost (or objective) function values summed accross all inputs, or (more or less equivalently) the average classification accuracy.

Say we have three datasets of similar size and content.  These correspond to slightly modified versions of those used in the last section [here](https://blbadger.github.io/nn-limitations.html), and are available in the github repository linked on this page above.

First there is the original training dataset,
![training](/neural_networks/train.png)

then there is the first test dataset,
![test 1](/neural_networks/test1.png)

and a second test set
![test 2](/neural_networks/test_2.png)

These three datasets are similar in appearance, and are actually images of cells from three different fruit fly brains.  There is a clear pattern between the labels 'Snap29' and 'Control': there are numerous blobs inside the cells of the former but not the latter.  

Now let's train the deep convolutional network on each dataset, using the other two datasets as the test sets for observation of training efficacy. The results are found [here](https://github.com/blbadger/blbadger.github.io/blob/master/neural_networks/nn_training).

The training results also display some more unexpected features. Firstly, one set of initial weights and biases can yield very different test classification accuracy: for example, the following array corresponds to the training accuracies achieved for one set of initial weights and biases:

```python
[0.94158554, 0.9137691, 0.7301808, 0.8706537, 0.94297636, 0.86369956, 0.4993046, 0.4993046, 0.7162726, 0.9179416]
```

And as expected, there is some variation between average test set classification accuracy depending on the initial weights and biases.

The test set classification accuracies for all three datasets are as follows:

![training results](/neural_networks/nn_training.png)

The median test classification accuracy varies widely depending on which dataset is chosen for training, even though they appear to be very similar when viewed by eye. At the extremes, the original training dataset results in a median training accuracy of $~90 %$, wereas training with the third dataset yeilds a $~50 %$  median accuracy, no better than chance as this is a binary classification. 

### One-way training

More clearly, the situation in the last section is that a particular CNN in question when trained on dataset 1 has a high test accuracy when applied to dataset 2, whereas the same network when trained on dataset 2 experiences poor test accuracy on dataset 1.


### The importance of (pseudo)randomization

It is standard practice to perform pseudo-random transformations on the inputs and parameters of neural networks before and during training. Shuffling or random selection of training data, initialization of weights and biases on a Gaussian (or Poisson etc.) distribution, randomized neuron drouput, and most objective function optimization procedures like stochastic gradient descent or Adaptive moment estimation readily spring to mind when one considers current methods.

It is worth asking the question of why this is: why do all these procedures involve randomization?  Some experimentation will convince one that randomization leads to better performance: for example, failing to randomize the initial weights $w_0$ and biases $b_0$ often leads to poor training and test results.  But if randomization 'works', why is this the case?

To begin, let's consider gradient descent.

Stochastic gradient descent can be thought of as the foundation upon which most optimization procedures are built upon, and the online (one training example at a time, that is) algorithm is as follows:

Given a vector of a network's weights and biases $v_0$, an objective function $F$, a learning rate $\eta$, and shuffled dataset $\mathscr S$,

$$
v_{i+1} = v_i - \eta \nabla F_j(v) \forall j \in \mathscr S
$$

When $i$ is equal to the size of $\mathscr S$, one epoch is completed and the entire process is repeated until $F$ is approximately minimized (or some other constraint is met).  

The randomization step is that the dataset is shuffled.  Why take that set in the first place?  The objective function $F$ that we are trying to minimize is usually thought of as a multidimensional 'landscape', with gradient descent being similar to rolling a ball down a hill.

 ### Backpropegation and subspace exploration
 
Neural networks were studies long before they became feasible computational models, and one of the most significant breakthroughs that allowed for this transition was the discovery of backpropegation.

---backprop details---

Backpropegation can be thought of as an optimal graph traversal method (or a dynamic programming solution) that updates a network's weights and biases with the fewest number of computations possible.  The goal of training is to find a certain combination of weights and biases (and any other trainable parameters) that yield a small objective function, and one way this could be achieved is by simply trying all possible weights and biases.  That method is grossly infeasible, and even miniscule networks with neurons in the single digits are unable to try all possible (assuming 32 bit precision) combinations.

Methods such as dropout are similar to the process of adjusting weights and biases randomly, only the process occurs withing the backpropegation framework rather than outside it, which would be far less efficient.  







