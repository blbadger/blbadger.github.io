## Flexible and Interpretable Deep Learning

Relying on the ability of deep learning techniques to be sufficient to learn successive representations of an input, we explore extremely flexible yet interpretable neural networks.

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

To start with, we will use a small dataset and tailor our approach to that specific dataset before then developing a generalized approach that is capable of modeling any abitrary grid of values.  Say we were given the following training data:


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
$$

where $x$ is a (6-dimensional) vector containing the formatted inputs of the relevant features we want to use for predicting `Elapsed Time`, denoted $y$.  The parameters of $w, b$ for linear models usually by minimizing the mean squared error

$$
MSE = \frac{1}{m} \sum_i (\hat {y_i} - y_i)
$$

Other more sophisticated methods may be employed in a similar manner in order to generate a scalar $\hat y$ from $x$ via tuned values of parameter vectors $w$ and $b$.  Now consider what is necessary for our linear model to perform well: first, that the true data generating distribution is approximately linear, but most importantly for this page that the input vector $x$ is properly formatted to avoid missing values, differences in numerical scale, differences in distribution etc.

When one seeks to format an input vector, there are usually many decision that one has to make.  For example: what should we do with missing values: assign a placeholder, or simply remove the example from our training data? What should we do if the scale of one feature is orders of magnitude different than another?  Should we enforce normality if the distribution for each feature, and if so what mean and standard deviation should we apply? 
How should we deal with categorical inputs, as only numerical vectors are accepted in the linear model?  For categorical inputs with many possible options, should we perform dimensionality reduction to attempt to capture most of the  information present in a lower-dimensional form? If so, what is the minimum number of options we should use and what method should we employ: a neural network-based embedding, sparse matrix encoding, or something else?

These questions and more are all implicitly or explicitly addressed when one formats data for a model such as this: the resulting vector $x$ is composed of 'hand-designed' features.  But how do we know that we have made the optimal decisions for formatting each input?  If we have many options of how to format many features, it is in practice impossible to try each combination of possible formats and therefore we do not actually know which recipe will be optimal.

The approach of deep learning is to avoid hand-designed features and instead allow the model to learn the optimal method of representing the input as part of the training process.  A somewhat extreme example of this would be to simply pass a transformation (encoding) $f$ of the raw character input as our feature vector $x$. For the first example in our training data above, this equates to some function of the string as follows:


$$
x = f(1 \;	2015-02-06 \; 22:24:17 \;1845 \;3441	\;33 \;14 \;21 \;861 \;3218)
$$

There are a number of different functions $f$ may be used to transform our raw input into a form suitable for a deep learning program.  In the spirit of avoiding as much formatting as possible, we can assign $f$ to be a one-hot character encoding, and then flatten the resulting tensor if our model requires it. One-hot encodings take an input and transform this into a tensor of size $|n|$ where $n$ is the number of possible categories, where the position of the input among all options determines which element of the tensor is $1$ (the rest are zero).  For example, a one-hot encoding on the set of one-digit integers could be

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



It turns out that this method can be employed quite successfully.  First to appreciate why this is somewhat surprising, consider the functions that this network must learn.



### Weaknesses of recurrent neural net architectures



### Structured sequence input encoding 


### Interpretable deep learning

It has been claimed in the past that deep learning is somewhat of a 'black box' machine learning approach: powerful, but mysterious with respect to the approach's inner workings. For a clear example of a machine learning technique that is the opposite of this, take a multiple linear regression.

$$

$$

In some respect, training this approach is similar to training a neural network: there are weights and biases that must be adjusted in order to minimize a cost function.  But although we could train a linear regression model with gradient descent, it is more efficient to note that the gradient of the cost function in quadratic and therefore simply finding the point with 0 gradient algebraically allows us to minimize the cost function in one step.

Linear regression is, as the name implies, linear with respect to the parameters that we train.  This means that all parameters are additive and scaling 

$$

$$

which is why the model is simple to optimize.  But most importantly for transparency, this means that there are clear interpretations of any linear model.  Say we have a regression that has 10 variables and the model assigns a weight of 0.1 to the first variable, and 2.5 to the second. From only this information, we know that the same change in the second variable as the first will lead to a 25-times increase in the response prediction.

Say one were trying to predict how many rubber duckies were in a pool, and this prediction depended on 10 variables, the first two being how many rubber duck factories were nearby and the second how many people at the pool who like rubber ducks.  If the second variable has a larger weight than the first, it is more important for predicting the output. We can even say that for every person who brings a rubber duck, we will have 2.5 more ducks in the pool.

This contrived example is to illustrate the very natural way that linear models may be interpreted: they key idea here is that one can separate out information for each input because the model itself is additive, and furthermore that we can know what the model will predict when we change a variable because all linear models are scaling. 

Nonlinear models are not necessarily additive or scaling.  In our above example, this could mean that we have to examine all 10 variables at once to make any prediction, and furthermore that we have to know how many rubber ducks are in the pool to make a prediction of how many will be added by changing any of the variables.  

