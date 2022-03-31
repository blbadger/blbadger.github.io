## Flexible and Interpretable Deep Learning

Relying on the ability of deep learning to form representations of an input, we explore extremely flexible yet interpretable neural networks.

### Introduction and background

Deep learning here is defined as a type of machine learning in which the input is represented in successive layers, with each layer being a distributed representation of the previous.  This allows for the approximation of very complicated functions by 

Deep learning approaches are often touted for having greater accuracy than other machine learning approaches, and for certain problems this is indeed true.  Tasks such as visual object recognition, speech transcription or language translation are among those problems at which deep learning approaches are currently the most effective at among all machine learning techniques, and for some problems the efficacy of deep learning is far beyond what has been achieved by other methods.

The kinds of tasks mentioned in the last paragraph are sometimes called 'AI-complete', which is an analogy to the 'NP-complete' problems that are as hard as all other NP-complete problems with regards to computational complexity reduction. These tasks are quite difficult for traditional machine learning, but are often  approachable using deep learning methods, and indeed most research on the topic since the seminal papers of 2006 have been focused on the ability of deep learning to out-perform other methods on previously-unseen data.

This page focuses on a different but related benefit of deep learning: the ability to learn an arbitrary representation of the input, allowing one to use one  model for a very wide array of possible problems.  The interpretability, or the ability to understand how the model arrives at any prediction, of this flexible approach is then explored.

### Abstract sequence modeling

Deep learning approaches such as neural networks are able to learn a wide array of functions (although there are certainly limits to what any machine learning approach is able to approximate, see [here](https://blbadger.github.io/nn-limitations.html) for more details

Work on this topic was in part inspired by the finding that the large language model GPT-3 was able to [learn arithmetic](https://openai.com/blog/grade-school-math/).  The research question: given that a large language model is able to learn simple functions on sequential character inputs, can a smaller model learn simple and complex functions using language-like inputs as well?  

The importance of this question is that if some model were to be able to learn how to approximate certain functions on an input comprising something as general as a sequence of characters, a machine learning practitioner would be able to avoid the often difficult process of formatting each feature for each dataset of interest. In the following sections, we shall see that the answer to this question is yes.

### Input Flexibility 

To start with, we will use a small dataset and tailor our approach to that specific dataset before then developing a generalized approach that is capable of modeling any abitrary grid of values.  Say we were given the following training data in the form of a design matrix:


|Market	|Order Made|	Store Number	|Cost |Total Deliverers	|Busy Deliverers	|Total Orders	|Estimated Transit Time	|Linear Estimation	|Elapsed Time|
| ----- | ---------| -------------- | --- | --------------- | --------------- | ----------- | --------------------- | ----------------- | ---------- |
|1	|2015-02-06 22:24:17	|1845	|3441	|33	|14	|21	|861	|3218	|3779|
|2	|2015-02-10 21:49:25	|5477	|1900	|1	|2	|2	|690	|2818	|4024|
|3	|2015-01-22 20:39:28	|5477	|1900	|1	|0	|0	|690	|3090	|1781|
|3	|2015-02-03 21:21:45	|5477	|6900	|1	|1	|2	|289	|2623	|3075|

We have a number of different inputs and the task is to predict **Elapsed Time** necessary for a delivery. The inputs are mostly integers except for the **Order Made** feature, which is a timestamp.  The names of the third-to-last and second-to-last columns indicate that we are likely developing a boosted model, or in other words a model that takes as its input the outputs of another model.

For some perspective, how would we go about predicting the **Elapsed Time** value if we were to apply more traditional machine learning methods? As most features are numerical and we want an integer output, the classical approach would be to hand-format each feature, apply normalizations as necessary, feed these into a model, and recieve an output that is then used to train the model.  This was how the **Linear Estimation** column was made: a multiple ordinary least squares regression algorithm was applied to a selection of formatted and normalized features with the desired output.  More precisely, we have fitted weight $w \in \Bbb R^n$ and bias (aka intercept) $b \in \Bbb R^n$ vectors such that the predicted value $\hat {y}$ is

$$
\hat {y} = w^Tx + b
\tag{1}
$$

where $x$ is a (6-dimensional) vector containing the formatted inputs of the relevant features we want to use for predicting **Elapsed Time**, denoted $y$.  The parameters of $w, b$ for linear models usually by minimizing the mean squared error

$$
MSE = \frac{1}{m} \sum_i (\hat {y_i} - y_i)
$$

Other more sophisticated methods may be employed in a similar manner in order to generate a scalar $\hat y$ from $x$ via tuned values of parameter vectors $w$ and $b$.  Now consider what is necessary for our linear model to perform well: first, that the true data generating distribution is approximately linear, but most importantly for this page that the input vector $x$ is properly formatted to avoid missing values, differences in numerical scale, differences in distribution etc.

When one seeks to format an input vector, there are usually many decision that one has to make.  For example: what should we do with missing values: assign a placeholder, or simply remove the example from our training data? What should we do if the scale of one feature is orders of magnitude different than another?  Should we enforce normality if the distribution for each feature, and if so what mean and standard deviation should we apply? 
How should we deal with categorical inputs, as only numerical vectors are accepted in the linear model?  For categorical inputs with many possible options, should we perform dimensionality reduction to attempt to capture most of the  information present in a lower-dimensional form? If so, what is the minimum number of options we should use and what method should we employ: a neural network-based embedding, sparse matrix encoding, or something else?

These questions and more are all implicitly or explicitly addressed when one formats data for a model such as this: the resulting vector $x$ is composed of 'hand-designed' features.  But how do we know that we have made the optimal decisions for formatting each input?  If we have many options of how to format many features, it is in practice impossible to try each combination of possible formats and therefore we do not actually know which recipe will be optimal.

The approach of deep learning is to avoid hand-designed features and instead allow the model to learn the optimal method of representing the input as part of the training process.  A somewhat extreme example of this would be to simply pass a transformation (encoding) $f$ of the raw character input as our feature vector $x$. For the first example in our training data above, this equates to some function of the string as follows:


$$
x = f(1 \;2015-02-06 \; 22:24:17 \;1845 \;3441	\;33 \;14 \;21 \;861 \;3218)
$$

There are a number of different functions $f$ may be used to transform our raw input into a form suitable for a deep learning program.  In the spirit of avoiding as much formatting as possible, we can assign $f$ to be a one-hot character encoding, and then flatten the resulting tensor if our model requires it. One-hot encodings take an input and transform this into a tensor of size $\vert n \vert$ where $n$ is the number of possible categories, where the position of the input among all options determines which element of the tensor is $1$ (the rest are zero).  For example, a one-hot encoding on the set of one-digit integers could be

$$
f(0) = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] \\
f(1) = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] \\
f(2) = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] \\
\vdots \\
f(9) = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
$$

Note that it does not matter which specific encoding method we use, and that any order of assignment will suffice (for example, we could swap $f(0)$ and $f(1)$).  Nor does it particularly matter that we place a $1$ in the appropriate position for our tensor: any constant $c$ would work, as long as this constant is not changed between inputs.  What **is** important is that each category of input is linearly independent of all other categories, meaning tat we cannot apply some compression such as

$$
f(0) = [1, 0] \\
f(1) = [0, 1] \\
f(2) = [1, 1] \\
\vdots
$$

or even simply assign each value to a single tensor,

$$
f(0) = [0] \\
f(1) = [1] \\
f(2) = [2] \\
\vdots
$$

These and any other methods of encoding such that each input category is no longer linearly independent of each other (and not normed to one constant $c$) are generally bad because it introduces information about the input $x$ that is not likely to be true: if we were encoding alphabetical characters instead of integers, each character should be 'as different' from each other character but this would not be the case if we applied a linearly dependent encoding.

For some perspective, we can compare this input method to a more traditional approach that is currently used for deep learning.  The strategy for that approach is as follows: inputs that are low-dimension may be supplied as one-hot encodings (or as basis vectors of another linearly independent vector space) and inputs that are high-dimensional (such as words in the english language) are first embedded using a neural network, and the result is then given as an input for the chosen deep learning implementation.

Embedding is the process of attempting to recapitulate the input on the output, while using less information that originally given. The resulting embedding, also called a latent space, is no longer generally linearly independent but instead contains information about the input itself: for example, vectors for the words 'dog' and 'cat' would be expected to be more similar to each other than the vectors 'dog' and 'tree'.  

Such embeddings would certainly be possible for our character-based approach, and may even be optimal, but the following intuition may explain why they are not necessary for successful application of our sequential encoding: we know that any deep learning model will reduce dimensionality to 0 (in the case of regression) or $n$ (for $n$ categories) depending on the output, so we can expect the model itself to perform the same tasks that an embedding would.  This is perhaps less efficient than separating out the embedding before then training a model on a given task, but it turns out that this method can be employed quite successfully.  

To summarize, we can bypass not only the need for hand-chosen formatting and normalization procedures but also for specific embeddings when using a sequential character-based input approach.

### Weaknesses of recurrent neural net architectures

One choice remains: how to deal with missing features.  For models such as (1), this may be accomplished by simply removing training examples that exhibit missing features, and enforcing the validation or test datasets to have some value for all features. The same approach could be applied here, but such a strategy could lead to biased results.  What happens when not only a few but the majority of examples both for training and testing are missing data on a certain feature?  Removing all examples would eliminate the majority of information in our training dataset, and is clearly not the best way to proceed.

Perhaps the simplest way to proceed would be to assign our sequence-to-tensor mapping function $f$ to map an empty feature to an empty tensor.  For example, suppose we wanted to encode an $x_i$ which was missing the first feature value of $x_c$:

$$
x_c = f(1 \;	2015-02-06 \; 22:24:17 \;1845 \;3441	\;33 \;14 \;21 \;861 \;3218) \\
x_i = f( \;	2015-02-06 \; 22:24:17 \;1845 \;3441	\;33 \;14 \;21 \;861 \;3218)  
$$

in this case $x_c = x_i$ except for the first 10 elements of $x_c$, which $x_i$ omits.  The non-concatenated tensors would be

$$
x_c = [[0,1,0,0,0,0,0,0,0,0],[0,0,1,...] ...]\\
x_i = [[],[0,0,1,...] ...]
$$

which means that the concatenated versions begin as $[0,1,0...]$ for $x_c$ and $[0,0,1,...]$ for $x_i$, resulting in a variable-length $x$.

A specific class of deep learning models have been designed for variable-length inputs. These are recurrent neural networks, and they operate by examining each input sequentially (ie for $x_c$ the network would first take an input as $[0,1,0,0,0,0,0,0,0,0]$), using that input to update the activations for each neuron according the the weights $w$ and biases $b$ present in that network, and then perhaps generating an output which can be viewed as an observation of the activations of the final layer of neurons. 

For our regression problem, the only output that we would need to be concerned with is the output after the last input was recieved by the network. This output would then be used for loss back-propegation according to the gradient of the loss.

One difficulty with recurrent neural networks is that they may suffer from exploding gradients: if each input adds to the previous, so can the gradient from each output leading to exceptionally large gradients for long sequences and thus difficulties during stochastic gradient descent.  This particular problem may be remedied a number of different ways, one of which involves gating the activations for each sequential input.  Another problem is that because network activations are added together, pure recurrent networks often have difficulty 'remembering' information fed into the network at the start of the sequence whilst nearing the end.

Both of the problems in the last paragraph are addressed using the Long Short-term Memory recurrent architecture, which keeps both short-term as well as a long-term memory modules in the form of neurons that are activated and gated. However, even these and other modified recurrent nets do not address perhaps the most difficult problem of recurrence during training: the process has a tendancy to 'forget' older examples relative to non-recurrent architectures, as there is $n^2$ distance between the last update and first update with respect to parameter activations.  See [this page](https://blbadger.github.io/neural-networks2.html) for more informatino on this topic.

That said, recurrent neural net architectures can be very effective deep learning procedures.  But there is a significant drawback to using them for our case: we lose information on the identity of each feature.  To explain why this is, imagine observing the following tensor as the first input: $[0,1,0...]$. There is not way a priori for you to know which feature this input tensor corresponds to, whereas in our grid of values above, we certainly would know which column (and therefore which feature) an empty value was located inside. 

The importance of maintaining positional information for tasks such as these has been borne out by experiment: identical LSTM-based neural networks perform far better with respect to minimization of test error using input functions $f$ that retains positional information compared to $f$ that do not.

### Structured sequence input encoding 

How do we go about preserving positional information for a sequence in order to maintain feature identity for each input element?  A relatively simple but effective way of doing this is to fix the number of elements that are assigned to be a constant value $c$, and then to provide a place-holder value $v$ for however many elements that a feature is missing.  In our example above, this could be accomplished by adding an eleventh element to each tensor (which we can think of as denoting 'empty') and performing one-hot encoding using this expanded vocabulary,

$$
x_c = [[0,1,0,0,0,0,0,0,0,0,0],[0,0,1,...] ...]\\
x_i = [[0,0,0,0,0,0,0,0,0,0,1],[0,0,1,...] ...]
$$

The important thing here is to keep the element denoting 'empty' to be one that is rarely if ever used to denote anything else.  For example, if we were to use the tensor for $f(0)$ to denote an empty character, we would lose information if $0$ were found in any of the features because a priori the model cannot tell if the tensor $f(0)$ denotes a zero element or an empty element.

This structured input can be implemented as follows: first we import relevant libraries and then the class `Format` is specified.  Note that source code for this page may be found [here](https://github.com/blbadger/nnetworks/tree/master/interprets).

```python
# fcnet.py
# import standard libraries
import string
import time
import math
import random

# import third-party libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.utils import shuffle

import torch
import torch.nn as nn

class Format:

	def __init__(self, file, training=True):

		df = pd.read_csv(file)	
		df = df.applymap(lambda x: '' if str(x).lower() == 'nan' else x)
		df = df[:10000]
		length = len(df['Elapsed Time'])
		self.input_fields = ['Store Number', 
				'Market', 
				'Order Made',
				'Cost',
				'Total Deliverers', 
				'Busy Deliverers', 
				'Total Orders',
				'Estimated Transit Time',
				'Linear Estimation']

		if training:
			df = shuffle(df)
			df.reset_index(inplace=True)

			# 80/20 training/validation split
			split_i = int(length * 0.8)

			training = df[:][:split_i]
			self.training_inputs = training[self.input_fields]
			self.training_outputs = [i for i in training['positive_two'][:]]

			validation_size = length - split_i
			validation = df[:][split_i:split_i + validation_size]
			self.validation_inputs = validation[self.input_fields]
			self.validation_outputs = [i for i in validation['positive_two'][:]]
			self.validation_inputs = self.validation_inputs.reset_index()

		else:
			self.training_inputs = self.df # not actually training, but matches name for stringify
```
Then in this class we implement a method that converts each row of our design matrix into a sequence of characters, which will be used as an argument to $f()$.

```python
class Format:
  ...
  
  def stringify_input(self, index, training=True):
		"""
		Compose array of string versions of relevant information in self.df 
		Maintains a consistant structure to inputs regardless of missing values.

		Args:
			index: int, position of input

		Returns:
			array: string: str of values in the row of interest

		"""


		taken_ls = [4, 1, 15, 4, 4, 4, 4, 4, 4]

		string_arr = []
		if training:
			inputs = self.training_inputs.iloc[index]
		else:
			inputs = self.validation_inputs.iloc[index]

		fields_ls = self.input_fields
		for i, field in enumerate(fields_ls):
			entry = str(inputs[field])[:taken_ls[i]]
			while len(entry) < taken_ls[i]:
				entry += '_'
			string_arr.append(entry)

		string = ''.join(string_arr)
		return string
```
Now we implement another class method which will perform the task of $f()$, ie of converting a sequence of characters to a tensor.  In this particular example, we proceed with concatenating each character's tensor into one in the `tensor = tensor.flatten()` line because we are going to be feeding our inputs into a simple fully-connected feed-forward neural network architecture.

```python
	@classmethod
	def string_to_tensor(self, input_string):
		"""
		Convert a string into a tensor

		Args:
			string: str, input as a string

		Returns:
			tensor: torch.Tensor() object
		"""

		places_dict = {s:int(s) for s in '0123456789'}
		for i, char in enumerate('. -:_'):
			places_dict[char] = i + 10

		# vocab_size x batch_size x embedding dimension (ie input length)
		tensor_shape = (len(input_string), 1, 15) 
		tensor = torch.zeros(tensor_shape)

		for i, letter in enumerate(input_string):
			tensor[i][0][places_dict[letter]] = 1.

		tensor = tensor.flatten()
		return tensor 
```
and now we can assemble a neural network. Here we implement a relatively simple fully-connected network by inheriting from the `torch.nn.Module` library and specify a 5-layer (3 hidden layer) architecture.
 
 ```python
 class MultiLayerPerceptron(nn.Module):

	def __init__(self, input_size, output_size):

		super().__init__()
		self.input_size = input_size
		hidden1_size = 500
		hidden2_size = 100
		hidden3_size = 20
		self.input2hidden = nn.Linear(input_size, hidden1_size)
		self.hidden2hidden = nn.Linear(hidden1_size, hidden2_size)
		self.hidden2hidden2 = nn.Linear(hidden2_size, hidden3_size)
		self.hidden2output = nn.Linear(hidden3_size, output_size)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(0.3)

	def forward(self, input):
		"""
		Forward pass through network

		Args:
			input: torch.Tensor object of network input, size [n_letters * length]

		Return: 
			output: torch.Tensor object of size output_size

		"""

		out = self.input2hidden(input)
		out = self.relu(out)
		out = self.dropout(out)

		out = self.hidden2hidden(out)
		out = self.relu(out)
		out = self.dropout(out)

		out = self.hidden2hidden2(out)
		out = self.relu(out)
		out = self.dropout(out)

		output = self.hidden2output(out)
		return output
 ```
 
This architecture may be understood as accomplishing the following: the last layer is equivalent to a linear model $y=m^Tx' + b$ on the final hidden layer $x'$ as an input, meaning that the hidden layers are tasked with transforming the input vector $x$ into a representation $x'$ that is capable of being modeled by (1).

### Controls

Before applying our model to the dataset in question, we can directly assess the efficacy or our sequential character encoding method by applying the model to what in experimental science is known as a positive control.  This is a (usually synthetic) dataset in which the desired outcome is known beforehand, such that if the experimental apparatus were successful in being able to perform its function then we would be able to arrive at some specific output.  

In this case, we are interested in determining whether or not this relatively small neural network is capable of learning defined functions, in contrast to the as-yet undefined function that determines the actual delivery time (together with any Bayesian error). Our first control is as follows:

|Market	|Order Made|	Store Number	|Cost |Total Deliverers	|Busy Deliverers	|Total Orders	|Estimated Transit Time	|Linear Estimation	|Control Output|
| ----- | ---------| -------------- | --- | --------------- | --------------- | ----------- | --------------------- | ----------------- | ---------- |
|1	|2015-02-06 22:24:17	|1845	|3441	|33	|14	|21	|861	|3218	|330|
|2	|2015-02-10 21:49:25	|5477	|1900	|1	|2	|2	|690	|2818	|10|
|3	|2015-01-22 20:39:28	|5477	|1900	|1	|0	|0	|690	|3090	|10|
|3	|2015-02-03 21:21:45	|5477	|6900	|1	|1	|2	|289	|2623	|10|

where the function is

$$
y = 10d
$$

where $d$ is the **Total Deliverers** input. We can track test (previously unseen) accuracy during training on any regression by plotting the actual value agains the model's prediction: a perfect model will have all points aligned on the line $y=x$.  Plotting the results of the above model using our sequential input encoding detailed above, we have

{% include youtube.html id='KgCuK6v_MgI' %}

so clearly for this control function, the model arrives at near-perfect accuracy fairly quickly.  It is important here to note that this is a much more difficult problem for the model than at first it may seem because of the encoding method: rather than simply assign an appropriate weight to an input neuron encoding $d$ instead the model must learn to decode the encoding function, and then apply the appropriate weight to the decoded value.  In effect, we are enforcing a distributed representation of the input because the encoding method itself is distributed (here $4*15=60$ tensor values rather than $1$).

We can also experiment with the network learning a more complicated rule, this time the non-linear (with respect to the input) mapping

$$
y = (c/100) * b
$$

where $b$ is the **Busy Deliverers** input and $c$ is the **Cost** input field.  Once again, the model is capable of very accurate estimations on the test dataset:

{% include youtube.html id='rZRQa3ExzTU' %}

and the estimation accuracy for both positive controls diminishes as the number of training examples increases, which is some small experimental evidence suggests that $\sum_i \hat {y_i} - y_i \to 0$ as $i \to \infty$.

These examples show that defined functions on the input are capable of being approximated quite well by the sequential character encoding method. 

### Justification of sequence-based character encodings

We have seen that the above encoding approach is effective, but why is this? It seems that we are simply feeding in raw information to the network and then are more or less by luck meeting with success.  Happily there are some nice theoretical considerations as to why this method is successful.

Consider the network architecture above: in this case the input is a tensor of concatenated one-hot vectors, and the output after three hidden layers is a single neuron that performs a linear regression on the second-to-last layer.  Omitting most components, the architecture may be visualized as follows:

![deep learning architecture]({{https://blbadger.github.io}}neural_networks/nn_architecture.png)

Now let's compare this to the normal approach to encoding categorical inputs that may be of high dimensionality.  Rather than employ large one-hot vectors, a better way is to perform embeddings on each of the features in question.  This may be done with autoencoder neural networks as follows: first each feature is separated and then one hidden layer of smaller dimensionality than that input feature is trained to return the input as well as possible.  The hidden layers contain what are normally called embeddings, or encodings in lower dimension, of the input.  These hidden layers may then be used as the input layer of the deep learning model that will then produce an output.  For a visualization of this process see below.

![deep learning architecture]({{https://blbadger.github.io}}neural_networks/nn_embeddings.png)

now consider what happens when we feed the input directly into the model, bypassing the embedding stage.  If successful, each hidden layer of the model will learn on or more distributed representations of the input, meaning that the input will be represented accross the layer such that the final representation allows for the final neuron to perform a linear regression $\hat {y} = w^Tx + b$ and obtain an accurate result, regardless of the true function one attempts to model.

In the model above, it is clear that each successive representation of the input will be of lower dimension than the last because each layer contains fewer parameters than the last.  But in the general case, we can also say that between the input and output, there is a representation that is of lower dimension than the input itself (otherwise training would be impossible, see the last section of [this page](https://blbadger.github.io/nn-limitations.html) for more details).  Therefore every model for some hidden layer we obtain a representation of the input in a lower dimension, or equivalently an embedding.

Thus we are allowed to bypass an embedding stage because the model will be expected to obtain its own embeddings. 

![deep learning architecture]({{https://blbadger.github.io}}neural_networks/nn_including_embeddings.png)

The traditional approach of first training embeddings using autoencoders before proceeding to use those as inputs into the model is analagous to greedy layer-wise pretraining for the first layer of the network, with the objective function being a distance measurement from the input rather than the model's objective function.  

### Interpretable deep learning: Introduction

It has been claimed in the past that deep learning is somewhat of a 'black box' machine learning approach: powerful, but mysterious with respect to understanding how a model arrived at a prediction. For a clear example of a machine learning technique that is the opposite of this, let's revisit the linear regression.

$$
\hat {y} = w^Tx + b
\tag{1}
$$

In some respect, training this approach is similar to training a neural network: the weight $w$ bias $b$ vectors are adjusted in order to minimize a cost function.  Note that although we could train a linear regression model with gradient descent, it is more efficient to note that the gradient of the cost function in quadratic and therefore simply finding the point with 0 gradient algebraically allows us to minimize the cost function in one step.

Linear regression is, as the name implies, linear with respect to the parameters that we train.  This means that for the regression function $f$, all parameters $w_n \in {w}$ and $b_n \in {b}$ are additive

$$
f(w_1 + w_2) = f(w_1) + f(w_2) \\ 
$$

and scaling

$$
f(a*w_1) = a*f(w_1)
$$

which is why the model is simple to optimize (as the gradient of this model is quadratic, and thus has one extremal point that can be found via direct computation of where $\nabla_w w^Tx + b = 0$ and $\nabla_b w^Tx + b = 0$.  

But most importantly for transparency, this means that there are clear interpretations of any linear model.  Say we have a regression that has 10 variables and the model assigns a weight of 0.1 to the first variable, and 2.5 to the second. From only this information, we know that the same change in the second variable as the first will lead to a 25-times increase in the response prediction.  This follows directly from the additive (which means that we can consider each parameter separately) and scaling (such that constant terms may be taken outside a function mapping) nature of linear functions.

Say one were trying to predict how many rubber duckies were in a pool, and this prediction depended on 10 variables, the first two being how many rubber duck factories were nearby and the second how many people at the pool who like rubber ducks.  If the second variable has a larger weight than the first, it is more important for predicting the output. We can even say that for every person who brings a rubber duck, we will have 2.5 more ducks in the pool.

This contrived example is to illustrate the very natural way that linear models may be interpreted: they key idea here is that one can separate out information for each input because the model itself is additive, and furthermore that we can know what the model will predict when we change a variable because all linear models are scaling. 

Nonlinear models may be non-additive, non-scaling, or both non-additive and non-scaling.  In our above example, this could mean that we have to examine all 10 variables at once to make any prediction, and furthermore that we have to know how many rubber ducks are in the pool to make a prediction of how many will be added by changing any of the variables.

Thus nonlinear models by their nature are not suited for interpretations that assume linearity.  But we may still learn a great deal about how they arrive at the outputs they generate if we allow ourselves to explore further afield than linear interpretations.  One widely used class of method to interpret nonlinear models is called attribution, and it involves determining the most- and least-important parts of any given input.  

### Attribution via Occlusion

Given any nonlinear model that performs a function approximation from inputs to outputs, how can we determine which part of the input is most 'important'?  Importance is not really a mathematical idea, and is too vague a notion to proceed with.  Instead, we could say that more important inputs are those for which the output would be substantially different if those particular inputs were missing, and vice versa for less important inputs.  

This definition leads to a straightforward and natural method of model interpretation that we can apply to virtually any machine learning approach.  This is simply to first compute the output for some input, and then compare this output to a modified output that results from removing elements of the input one by one.  This technique is sometimes called a perturbation-style input attribution, because we are actively perturbing the input in order to observe the change in the output.  More commonly this method is known as 'occlusion', as it is similar to the process of a non-transparent object occluding (shading) light on some target.

There are a few different ways one can proceed with implementing occlusion. For a complete input map $x_c$ and a map of an input missing the first value ($x_i$), 

$$
x_c = f(1 \;	2015-02-06 \; 22:24:17 \;1845 \;3441	\;33 \;14 \;21 \;861 \;3218) \\
x_i = f( \;	2015-02-06 \; 22:24:17 \;1845 \;3441	\;33 \;14 \;21 \;861 \;3218)  
$$

recall the structured input method's approach to missing values maps

$$
x_c=[[0,1,0,0,0,0,0,0,0,0,0],[0,0,1,...]...] \\
x_i=[[0,0,0,0,0,0,0,0,0,0,1],[0,0,1,...]...]
$$

We generate an occluded input map $x_o$,

$$
x_o = f(\mathscr u\;	2015-02-06 \; 22:24:17 \;1845 \;3441	\;33 \;14 \;21 \;861 \;3218)  
$$

where $\mathscr u$ signifies an occlusion (here for the first character) in a similar way by simply zeroing out all elements of the first tensor, making the start of the occluded tensor as follows:

$$
x_c=[[0,1,0,0,0,0,0,0,0,0,0],[0,0,1,...]...] \\
x_o=[[0,0,0,0,0,0,0,0,0,0,0],[0,0,1,...]...]
$$

and then calculate the occlusion value $v$ by taking the difference between the model's mapping $m()$ of this changed input and the first:

$$
v = \vert m(x_c) - m(x_o) \vert
$$

where $x_o$ is our occluded input, as this unambiguously defines an occluded input that is, importantly, different than an empty input and also different from any normally used character input as well.  This difference is important because otherwise the occluded input would be indistinguishable from an input that had a truly empty first character, such that the network's output given this occluded input might fail to be different from the actual output if simply having an empty field or not were the most important information from that field.  Thus we could also use a special character to denote an occluded input, perhaps by enlarging the character tensor size to 12 such that the mapping is given as

$$
x_o=[[0,0,0,0,0,0,0,0,0,0,0,1],[0,0,1,...]...]
$$

Note that this method is easily applied to categorical outputs, in which case the occlusion value $v$ is

$$
v = \sum_i \vert m(x_c)_i - m(x_o)_i \vert
$$

where $i \in C$ denotes a category in the set of all possible output categories $C$.

Let's implement this occlusion for a regression output in a class `StaticInterpret` that is specifically tailored to the dataset shown earlier on this page (later there will be presented a version of this class that may be applied to any sequential input regardless of length). This class will take as arguments the neural network `model`, the set of torch.tensor objects `input_tensors` and another set of torch.tensor objects `output_tensors`.  As the input format (meaning which element corresponds to which input field) is unchanged for separate input tensors, we then assign the correct fields as `self.fields_ls` and the number of elements (characters) per field as `self.taken_ls`

```python
class StaticInterpret:

	def __init__(self, model, input_tensors, output_tensors):
		self.model = model 
		self.model.eval()
		self.output_tensors = output_tensors
		self.input_tensors = input_tensors
		self.fields_ls = ['Store Number', 
				'Market', 
				'Order Made',
				'Cost',
				'Total Deliverers', 
				'Busy Deliverers', 
				'Total Orders',
				'Estimated Transit Time',
				'Linear Estimation']
		self.embedding_dim = 15
		self.taken_ls = [4, 1, 15, 4, 4, 4, 4, 4, 4] # must match network inputs

```

Now we can define a class function that will determine the occlusion value.  This function takes a keyword argument `occlusion_size`, an integer that determines the number of sequential elements that are occluded simultaneously.  This parameter is useful as many input fields have multiple characters associated with them.  If we remove only one character at once, what happens to the output?  Perhaps nothing, if the other characters in that same field are able to provide sufficient information to the model.  At the same time, too large an occlusion leads to inaccuracies in attribution as one field's occlusion cannot be separated from anothers.  An occlusion of size 2 has experimentally been found to be generally effective for this kind of task.

```python
	def occlusion(self, input_tensor, occlusion_size=2):
		"""
		Generates a perturbation-type attribution using occlusion.

		Args:
			input_tensor: torch.Tensor
		kwargs:
			occlusion_size: int, number of sequential elements zeroed
					out at once.

		Returns:
			occlusion_arr: array[float] of scores per input index
		"""
		input_tensor = input_tensor.flatten()
		output_tensor = self.model(input_tensor)
		zeros_tensor = torch.zeros(input_tensor[:occlusion_size*self.embedding_dim].shape)
		total_index = 0
		occlusion_arr = [0 for i in range(sum(self.taken_ls))]
		i = 0
		while i in range(len(input_tensor)-(occlusion_size-1)*self.embedding_dim):
			# set all elements of a particular field to 0
			input_copy = torch.clone(input_tensor)
			input_copy[i:i + occlusion_size*self.embedding_dim] = zeros_tensor

			output_missing = self.model(input_copy)
			occlusion_val = abs(float(output_missing) - float(output_tensor))

			for j in range(occlusion_size):
				occlusion_arr[i // self.embedding_dim + j] += occlusion_val

			i += self.embedding_dim

		# max-normalize occlusions
		if max(occlusion_arr) != 0:
			correction_factor = 1 / (max(occlusion_arr))
			occlusion_arr = [i*correction_factor for i in occlusion_arr]

		return occlusion_arr
```

Prior to training, the attribution values would be expected to be more or less randomly assigned. When we observe a heatmap of the attribution value per sequence position (x-axis) given fifty different test examples (y-axis) we see that indeed our expectation is the case

![readable attribution]({{https://blbadger.github.io}}neural_networks/attributions_0000.png)

and see [the source code](https://github.com/blbadger/nnetworks/tree/master/interprets) for details on how to implement a heatmap-based visualization of model occlusion.

Using one of our controls, we may observe how a network 'learns' occlusion.  Recalling that this dataset's output is defined as

$$
y = 10d
$$

where $d$ is the **Total Deliverers** input, and observing that this input occupies the 24th through the 27th position (inclusive) in our input sequence (as there are fields of size 4, 1, 15, and 4 ahead of it and we are incrementing from 0 rather than 1), we expect the occlusiong attribution to be highest for these positions for the majority of inputs.  Note that the **Total Deliverers** field is closely related to the **Available Deliverers** and **Total Orders** fields that follow it, as an increase in one is likely to affect the other two.  In statistical terms, these fields are related via the underlying data generating distribution.

This means that although we may expect to see the **Total Deliverers** field as the most 'important' as measured using occlusion, a statistical model would be expected to also place fairly large attributions on the fields most similar to **Total Deliverers**.  And if we observe occlusion values per character during a training run (200 epochs), indeed we see that this is the case (note how positions 24-27 are usually among the brightest in the following heatmap after training).  

{% include youtube.html id='HcSUH0zTexQ' %}

### Saliency-based attribution using gradients

Occlusion is not the only way to obtain insight into which input elements are more or less important for a given output: another way we can learn this is to exploit the nature of the process of learning using a neural network or other similar deep learning models.  This particular method is applicable to any model that uses a differentiable objective function, and thus is unsuitable for most tree-based machine learning models but is happily applicable to neural networks.  Gradient-based input attribution is sometimes termed 'saliency'.

The principle behind gradient based approaches to input attribution for neural network models is as follows: given some input, we can compute the model output and also the gradient of that output with respect to the weights and biases of the model.  This gradient may then be back-propegated to the input layer of the network, as if the input were a parametric component of the neural network like the other layers. We then assess how the input contributes to the output's gradient by multiplying the input gradient by the input itself.

$$
g = \nabla_i f(x) * x
$$

where $\hat{y} = f(x)$ is the model output, $x$ is the model input, $i$ is the input layer, and $g$ is the gradientxinput.  Note that $\nabla_i f(x)$ and $x$ are both tensors, and as we want a tensor of the same size we use the Hadamard (element-wise) product, which when using `torch.Tensor` objects `t1, t2`, the Hadamard product may be obtained as `t1 * t2`.

The intuition behind this method is a little confusing for the researcher or engineer normally used to finding the gradient of an objective function with respect to a tunable parameter in a model, so before proceeding further let's note the differences between this gradient used to interpret an input and the gradient used to train a model.  

To summarize: the method by which the gradient of the objective (loss) function $J$ of the output given model a configuration $\theta$ denoted as $O(\theta)$ evaluated with respect to parameter $p$ is

$$
g = \nabla_p J(O(\theta))
$$

and we can imagine a landscape of this loss function with respect to one variable as a Cartesian graph.  The opposite of the gradient tells us the direction (and partial magnitude) with which $p$ should be changed in order to minimize $J(O(\theta))$, which can be visualized as follows

![loss gradient]({{https://blbadger.github.io}}neural_networks/loss_gradient.png)

For multiple variables, we can imagine this visualization as simply extending to multiple dimensions, with each parameter forming a basis in $\Bbb R^n$ space.  The gradient's component along the parameter's axis forms the direction (with some magnitude information as well) in which that parameter should be changed.

Now instead of evaluating the gradient with respect to the objective function, to determine how the output alone changes we can evaluate the gradient of the output $O(\theta)$ with respect to parameter $p$

$$
g = \nabla_p O(\theta)
$$

which can be visualized as

![output gradient]({{https://blbadger.github.io}}neural_networks/output1_gradient.png)

now to determine which parameter is more important for determining a given output, we can compare the gradients $g_1$ and $g_2$ that are computed with respect to $p_1$ and $p_2$, respectively

$$
g_1 = \nabla_{p_1} O(\theta) \\
g_2 = \nabla_{p_2} O(\theta)
$$

These can be visualized without resorting to multidimensional space as separate curves on a two dimensional plane (for any single value of $p_1$ and $p_2$ of interest) as follows:

![output gradient]({{https://blbadger.github.io}}neural_networks/output2_gradient.png)

Where the red arrows denote $-g_1$ and $-g_2$ as gradients point in the direction of steepest ascent.  In this particular example, $\vert g_1 \vert > \vert g_2 \vert$ as the slope of the tangent line is larger along the output curve for $p_1$ than along $p_2$ for the points of interest.  In multidimensional space, we would be comparing the gradient vector's components, where each parameter $p_n$ forms a basis vector of the space and the component is a projection of the gradient vector onto each basis.

The next step is to note that as we are most interested in finding how the output changes with respect to the *input* rather than with respect to the model's true parameters, we treat the input itself as a parameter and back-propegate the gradient all the way to the input layer (which is not normally done as the input does not contain any parameters as normally defined).  Thus in our examples, $(p_1, p_2) = (i_1, i_2)$ where $i_1$ is the first input and $i_2$ is the second.

Now that we have the gradient of the output with respect to the input, this gradient can be applied in some way to the input values themselves in order to determine the relation of the input gradient with the particular input of interest.  The classic way to accomplish this is to use Hadamard multiplication, in which each vector element of the first multiplicand $v$ is multiplied by its corresponding component in the second multiplicand $s$

$$
v = (v_1, v_2, v_3,...) \\
s = (s_1, s_2, s_3,...) \\
v * s= (v_1s_1, v_2s_2, v_3s_3, ...)
$$

and all together, we have

$$
g = \nabla_i f(x) * x
$$

This final multiplication step is perhaps the least well-motivated portion of the whole procedure.  When applied to image data in which each pixel (or in other words each input) can be expected to have a non-zero value, we would not expect to lose much information with Hadamard multiplication and furhtermore the process is fairly intuitive: brighter pixels (ie those with larger input values) that have the same gradient values as dimmer pixels are considered to be more important.  But with one-hot encodings where most input values are indeed 0, it is less clear as to whether or not this method necessarily does not lose information.  Empirically, for low-dimensional (<500 or so) tensors this is not the case.

Thus we see that what occurs during the gradientxinput calculation is somewhat similar to that for the occlusion calculation, except instead of the input being perrured in some way, here we are using the gradient to find an expectation of how the output should change if one were to perturb the input in a particular way.

The following class method implements gradientxinput:

```python
	def gradientxinput(self, input_tensor):
		"""
		 Compute a gradientxinput attribution score

		 Args:
		 	input: torch.Tensor() object of input
		 	model: Transformer() class object, trained neural network

		 Returns:
		 	gradientxinput: arr[float] of input attributions

		"""
		# enforce the input tensor to be assigned a gradient
		input_tensor.requires_grad = True
		output = self.model.forward(input_tensor)

		# only scalars may be assigned a gradient
		output_shape = 1
		output = output.reshape(1, output_shape).sum()

		# backpropegate output gradient to input
		output.backward(retain_graph=True)

		# compute gradient x input
		final = torch.abs(input_tensor.grad) * input_tensor
		saliency_arr = []
		s = 0
		for i in range(len(final)):
			if i % self.embedding_dim == 0 and i > 0: 
				saliency_arr.append(s)
				s = 0
			s += float(final[i])
		saliency_arr.append(s)

		# max norm
		maximum = 0
		for i in range(len(saliency_arr)):
			maximum = max(saliency_arr[i], maximum)

		# prevent a divide by zero error
		if maximum != 0:
			for i in range(len(saliency_arr)):
				saliency_arr[i] /= maximum

		return saliency_arr
```

which results in

{% include youtube.html id='2V3B3tuc6mY' %}

### Combining Occlusion and Gradientxinput

One criticism of pure gradient-based approaches such as gradientxinput is that they only apply to local information.  This means that when we compute

$$
\nabla_i f(x)
$$

we only learn what happens when $i$ is changed by a very small amount $\epsilon$, which follows from the definition of a gradient.  Techniques that attempt to overcome this locality challenge include integrated gradients and other such additive measures.  But those methods have their own drawbacks (integrated gradients involve modifying the input such that the gradient-based saliency may exhibit dubious relevance to the original).  

Happily, we already have a locality-free (non-gradient) based approach to attribution that we can add to gradientxinput in order to overcome this limitation. The intuition behind why occlusion is non-local is as follows: there is no a priori reason as to why a modification to the input at any position should yield an occluded input that is nearly indistinguishable from the original, and thus any results obtained from comparing the two images using our model do not apply only to the first image. Furthermore, the `occlusion_size` parameter provides a clear hyperparameter to prevent any incedental locality, as simply increasing this parameter size is clearly sufficient to increase the distance between $x_o$ and $x_c$ in the relevant latent space.

Thus it is not entirely inaccurate to think of gradientxinput as a continuous version of occlusion, as the former tells us what happens if we were to change the input by a very small amount whereas the latter tells us what would happen if we changed it by a large discrete amount. 

We may combine occlusion with gradientxinput by simply averaging the values obtained at each position, and the result is as follows:

{% include youtube.html id='-M15BxfmFRQ' %}

One may want to be able to have a more readable attribution than is obtained using the heatmaps above,  This may be accomplished by averaging the attribution value of each character over a number of input examples, and then aggregating these values (using maximums or averages or another function) into single measurements for each field. An abbreviated implementation (lacking normalization and aggregation method choice) of this method is as follows:

```python
	def readable_interpretation(self, count, n_observed=100, method='combined', normalized=True, aggregation='average'):
		"""
		...
		"""
		attributions_arr = []

		for i in range(n_observed):
			input_tensor = self.input_tensors[i]
			if method == 'combined':
				occlusion = self.occlusion(input_tensor)
				gradxinput = self.gradientxinput(input_tensor)
				attribution = [(i+j)/2 for i, j in zip(occlusion, gradxinput)]
			...
			attributions_arr.append(attribution)

		average_arr = []
		for i in range(len(attributions_arr[0])):
			average = sum([attribute[i] for attribute in attributions_arr])
			average_arr.append(average)
		...
		if aggregation == 'average':
			final_arr = []
			index = 0
			for i, field in zip(self.taken_ls, self.fields_ls):
				sum_val = 0
				for k in range(index, index+i):
					sum_val += average_arr[k]
				final_arr.append([field, sum_val/k])
				index += i
		...
		plt.barh(self.fields_ls, final_arr, color=colors, edgecolor='black')
		...
		return
```

(note that the working version may be found in the [source code](https://github.com/blbadger/nnetworks/tree/master/interprets)

For the trained network shown above, this yields

![readable attribution]({{https://blbadger.github.io}}neural_networks/readable_1.png)

and the correct input is attributed to be the most important, so our positive control indicates success.  What about a slightly more realistic function in which multiple input fields contribute to the output? Recall that in our nonlinear control function, the output $y$ is determined by the function

$$
y = (c/100) * b
$$

where $b$ is the **Busy Deliverers** input and $c$ is the **Cost** input.  During training we can see that the expected (20-23 and 28-31 inclusive) positions are the ones that obtain the highest attribution values for most inputs:

{% include youtube.html id='ARTrheoeXEI' %}

and when the average attribution per input is calculated, we see that indeed the correct fields contain the highest attribution.

![readable attribution]({{https://blbadger.github.io}}neural_networks/readable_nonlinear.png)

Now we can apply this method to our original problem of finding which input are important for predicting a delivery time.





