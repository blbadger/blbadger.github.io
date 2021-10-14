## Neural Networks II: 

This page is a continuation from from [part one](https://blbadger.github.io/neural-networks.html). Code for this page is found in [this Github repo](https://github.com/blbadger/nnetworks).

## The importance of (pseudo)randomization

It is standard practice to perform pseudo-random transformations on the inputs and parameters of neural networks before and during training. Shuffling or random selection of training data, initialization of weights and biases on a Gaussian (or Poisson etc.) distribution, randomized neuron drouput, and most objective function optimization procedures like stochastic gradient descent or Adaptive moment estimation readily spring to mind when one considers current methods.

It is worth asking the question of why this is: why do all these procedures involve randomization?  Each case is considered in the following sections.

### Initialization

Some experimentation will convince one that randomization leads to better performance: for example, failing to randomize the initial weights $w_0$ and biases $b_0$ often leads to poor training and test results.  But if randomization 'works', why is this the case?

### Stochastic Gradient Descent

To begin, let's consider gradient descent.

Stochastic gradient descent can be thought of as the foundation upon which most optimization procedures are built upon, and the online (one training example at a time, that is) algorithm is as follows:

Given a vector of a network's weights and biases $v_0$, an objective function $F$, a learning rate $\eta$, and shuffled dataset $\mathscr S$,

$$
v_{i+1} = v_i - \eta \nabla F_j(v_i) \\
\forall j \in \mathscr S
\tag{1}
$$

When $i$ is equal to the size of $\mathscr S$, one training epoch is completed and the entire process is repeated until $F$ is approximately minimized (or some other constraint is met).  

The randomization step is that the dataset is shuffled.  Why take that set in the first place?  The objective function $F$ that we are trying to minimize is usually thought of as a multidimensional 'landscape', with gradient descent being similar to finding a valley while looking at the ground under one's feet while walking.
The local gradient, if sampled repeatedly, provides the necessary information to find the approximate minimum point if the objective function is smooth enough.  

If the objective function is not smooth, gradient descent may result in the finding of a local minima that is not the true global minimum.  To prevent this from happening, gradient descent may be modified to include parameters such as momentum, or extended into optimizers like RMSprop or AdaGrad.  

Now back to the original question: why does stochastic gradient descent work best if it is stochastic?  Why not enumerate all training examples and iterate through them in order, updating the network via gradient descent using $1$?  If each training example is fed to the network during one epoch, from an informational perspective would it matter whether or not one example comes before another?


### Training memory

This section will require some familiarity with [recurrent neural networks](https://en.wikipedia.org/wiki/Recurrent_neural_network), which will be given briefly here.  Recurrent neural networks are a class of feedforward neural networks that are updated sequentially as time passes.  The simplest form, a fully recurrent network, has one hidden layer that is updated at each sequential element.  For example, a text classification network performing sentiment analysis (finding if a statement expresses positive or negative emotions) of the following sentance:

"The dog went to the store."

would simply iterate through the sentance ('The' then 'dog' then 'went' etc.), and at each word the activation in each hidden neuron would be updated.  At the end of the sentance, the hope is that some information from early elements is retained in the hidden layer's activations, as these are then used to generate some output.  Stochastic gradient descent (or some other optimization procedure) and backpropegation are then performed on the network, updating the network's weights and biases in accordance with the output that was a result of the activations 

Unfortunately, in practice recurrent neural networks suffer from the same problems as very deep networks of all kinds: unstable gradients.  

Now consider the process of training a non-recurrent neural network, perhaps a convolutional net performing image classification.  The first training example (or batch of examples) is fed to the network with initial weights and biases (represented as $v_0$), resulting in some output $o_1$.  The output is then input into the objective function, and the gradient wrt output is calculated before backpropegation is used to update the network's weights and biases to $v_1$ via $(1)$.  The process is then repeated with the second (batch) example, where the network's weights and biases configuration $v_1$ generates an output $o_1$ which then leads to updating $v_1 \to v_2$ and so on.

Therefore the sequence of weight and bias configurations is

$$
(v_0, o_0), (v_1, o_1), (v_2, o_2), \; \cdots \; , (v_n, o_n)
$$

Now note that the final configuration $v_n$ depends not only on the prior configuration $v_{n-1}$ but also on all previous configurations and outputs as follows:

$$
v_0 + i_0 \to o_0 \to v_1 \\
v_1 + i_1 \to o_1 \to v_2 \\
\vdots \\
v_{n-1} + i_n \to o_n \to v_n
$$

Therefore successive configuration and outputs form a directed graph 

$$
v_0 \to o_0 \to v_1 \to o_1 \to \cdots \to v_n 
$$

But if we replace 'node' with 'configuration', the definition of a recurrent neural network is arrived at. This means that the final configuration $v_n$ can be thought of as retaining a 'memory' of $v_0, v_1...$ in that these weight and bias configurations were updated by future outputs, but not erased by them.  Instead of updating neuronal activations for each element in an input sample (as occurs in traditional rNNs), the configuration weights and biases are updated for each sample in an epoch (and activations do not persist between samples).  But a network's configuration determines its activations, such that the same result of memory over time exists.

To summarize, training any neural network is not instantaneous but instead occurs over a certain time interval, and therefore may be thought of as a dynamical system.  In this system, the final network configuration $v_n$ depends not only on the network inputs but also which input is fed to the network in what order.  

Considered carefully, therefore, any network network undergoing training via updating weights and biases over time is a type of recurrent neural network, with training states $v_0, v_1, ... , v_n$ forming a directed graph along a temporal sequence across training examples (rather than across a elements of one input, as is the case for rnns). 

It may be beneficial, therefore, to develop a theory of extending memory gates similar to those used in LSTMs and GRUs to the training of any network, such that a network's configuration at any point $v_m$ is influenced not only by $v_{m-1}$ but also $v_{m-2}$.  

### Input randomization and training memory

It was asserted in the previous section that the sequence of neural network configurations, inputs, and outputs form a directed, acyclic graph during training.  Because the graph is directed, there is no guarantee that reversing the order of $i_0, i_1, i_2, ..., i_{n-1}$ (and thus the order of outputs too) would result in the same sequence of configurations $v_0, v_1, v_2$.  This is to say that the order of training examples matters, such that the ability to minimize an objective function in the final configuration $v_n$ will vary depending on the order of training examples provided because the graph path taken is changed.

Now imagine training for one epoch only, meaning that all training examples are fed to the network exactly once (singly or in batch form).  What happens if we first group the training examples by similarity, perhaps using latent space vectorization, and then feed the resulting sequence to the network?  For simplicity, suppose there are inputs two types (in equal number), those belonging to set $\mathscr A$ and other belonging to set $\mathscr B$ such that all $a \in \mathscr A$ are seen by the network before $b \in \mathscr B$.  What happens to $v_n$?

Assuming that the network's hyperparameters (non-trainiable variables) are not changed during training, $v_n$ will reflect $\mathscr B$ moreso than $\mathscr A$, for the simple reason that the configuration updates for $a \in \mathscr A$ are 'overwritten' by $b \in \mathscr B$.  More precisely, the minimum distance between $v_n$ and an earlier configuration $v_m$ resulting from an update from either set is much greater for $a$ than for $b$: there are $\lvert \mathscr B \rvert$ nodes (updates) between $v_{ma}$ and $v_{mb}$.  

This is to say that the network after training would be expected to classify $a \in \mathscr A$ worse than $b \in \mathscr B$.  Instead, alternating $a, b, a, b, ...$ results in a distance $v_m(a) - v_m(b) = 1$.  Without knowing a priori which elements of the training set belong to $\mathscr A$ or $\mathscr B$, simply shuffling the training set minimized the expected value of $v_{ma} - v_{mb}$ and would therefore present the best solution to the problem of recency in training memory.

An example: 

Suppose one were to attempt to infer the next letter in a word using a character level recurrent neural net, and that all inputs were slight variations on the phrase 

$$
abababab
$$

and the network was trained on the task of returning the final letter. Assuming the training was effective, the expected output given an input 

$$
abababa
$$

would be $b$.  Indeed, relatively simple tasks such as these are well-accomplished by recurrent neural nets with no fancy gating.  But now suppose that a non-recurrent (in the traditional sense) neural network were trained on a sequence of $a \in \mathscr A$ and $b \in \mathscr B$ as above.  The expected next output given sequential alternating inputs

$$
a, \; b, \; a, \; b, \; a ,..., \;a
$$

would be $b$. If the above argument (that any network while training over many inputs behaves as a recurrent neural net for one input) is accepted, the alternation in input type contributes to $v_n$ minimizing its objective function.

Thus although alternating input types minimized the value of $v_{ma} - v_{mb}$, this alternation would be expected to cause the network to 'learn' that output type is alternating, and if this were not general to a test set then accuracy would necessarily follow. 

### Why randomize inputs during or between epochs

Randomization of inputs between or during training epochs is also effective for minimizing objective function loss. Why is this the case: once training inputs have been randomized, what further purpose would shuffling the inputs again serve?  

In the above sections, it was claimed that a neural net configuration $v_i$ depends not only on the inputs, but on the sequence of those inputs.  In dynamical system terms, a periodic training input 'trains' the network to expect periodic inputs in a test set regardless of whether or not the input is periodic. 

Suppose we have three elements per training set, and we want to perform three epochs of training.  After randomizing the inputs, there is

$$
a, c, b
$$

such that the full training session is

$$
a, c, b, a, c, b, a, c, b
$$

But this again is a periodic sequence, with a periodicity being the size of the dataset.  Over many epochs, this periodicity becomes reflected in the final network configuration $v_n$ just as a shorter period would be.  

### A possible benefit from randomized inputs during training

Returning to the successive configurations $v_0, v_1, ...$ the network takes over time during training, the directed graph representing these configurations given inputs and outputs is as follows:

$$
v_0 + i_0 \to o_0 \to v_1 + i_1 \to o_1 \to \cdots \to v_n 
$$

Now suppose one epoch is completed, and $v_n$ has decreased the objective function's output but not to the global minimum.  If one wishes to begin another training epoch and further minimize the objective function's output, what sequence of inputs should be used?

One option would be to use the same sequence of inputs as for the first epoch, that is, $i_0, i_1, i_2, i_3, ..., i_{n-1}$.  Now the initial configuration for the second epoch is not expected to be the same as it was before the first epoch, or $v_{02} \neq v_{0}$ and therefore the sequence $o_{02}, o_{12}, o_{22}, ..., o_{n2}$ would not be the same either.  But it would be more similar than if $i_0, i_1, i_2, ...$ were reordered.

To see why this is, observe that we can assign a vector to the input sequence $I = i_0, i_1, i_2, ... i_{n-1}$.  There are many vector additions and multiplications and other operations involved in updating $v_0 \to v_n$, but we can combine all of these into one operation $\circ$ to give $v_0 \circ I = v_n$.  

Finally, is it very likely that the ideal path from $v_{00} \to v_{nn}$ such that $v_{nn}$ minimized the objective function $F$ was achieved using the initial ordering $i_0, i_1, i_2 ... i_{n-1}$?  No, given that there are $(n-1)!$ ways of ordering $n-1$ inputs, without prior knowledge then the chance of choosing the best initial path is $1/(n-1)!$ assuming each path is unique.  Reordering the input sequence between epochs increases the chances of choosing a better path for one epoch (as well as a worse path).  

If many paths are of similar 'quality', but some paths are much better, then reordering the input sequence can act to minimize the objective function in $v_n$ by increasing the chances of landing on a better path.

### Backpropegation and subspace exploration
 
Neural networks were studies long before they became feasible computational models, and one of the most significant breakthroughs that allowed for this transition was the development of backpropegation.

Backpropegation can be thought of as an optimal graph traversal method (or a dynamic programming solution) that updates a network's weights and biases with the fewest number of computations possible.  The goal of training is to find a certain combination of weights and biases (and any other trainable parameters) that yield a small objective function, and one way this could be achieved is by simply trying all possible weights and biases.  That method is grossly infeasible, and even miniscule networks with neurons in the single digits are unable to try all possible (assuming 32 bit precision) combinations.


### Non-commutative training and testing

From the theoretical considerations presented [here](https://blbadger.github.io/nn-limitations.html) and elsewhere, it is clear that one inherent limitation in neural nets is the propensity for nearly indistinguishable (with respect to the feature space) inputs to be mapped to very different outputs.  Input examples such as these are called 'adversarial examples'. 

The collection of images used to train a network may be thought of as a single input (broken up into smaller pieces for convenience).  The theorem referenced above is not particular to single input images but instead applies to any input, leading to the following question: could nearly indistinguishable training datasets yield very different outputs?  Here outputs may be the cost (or objective) function values summed accross all inputs, or (more or less equivalently) the average classification accuracy.

Say we have three datasets of similar size and content.  These correspond to slightly modified versions of those used in the last section [here](https://blbadger.github.io/nn-limitations.html), and are available in the github repository linked on this page above.

First there is the original training dataset,
![training](/neural_networks/train.png)

then there is a first test dataset,
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

More clearly, the situation in the last section is that a particular CNN in question when trained on dataset 1 has a high test accuracy when applied to dataset 2, whereas the same network when trained on dataset 2 experiences poor test accuracy on dataset 1.  







