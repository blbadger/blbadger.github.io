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

### Input flexibility and the weaknesses of recurrent neural net architectures

|Market	|Order Made|	Store Number	|Cost	Total Deliverers	|Busy Deliverers	|Total Orders	|Estimated Transit Time	|Linear Estimation	|Elapsed Time|
| ----- | ---------| -------------- | ----------------------| --------------- | ----------- | --------------------- | ----------------- | ---------- |
|1	|2015-02-06 22:24:17	|1845	|3441	|33	|14	|21	|861	|3218	|3779|
|2	|2015-02-10 21:49:25	|5477	|1900	|1	|2	|2	|690	|2818	|4024|
|3	|2015-01-22 20:39:28	|5477	|1900	|1	|0	|0	|690	|3090	|1781|
|3	|2015-02-03 21:21:45	|5477	|6900	|1	|1	|2	|289	|2623	|3075|


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

