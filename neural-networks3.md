## Interpretable Deep Learning

Relying on the ability of deep learning techniques to be sufficient to learn successive representations of an input, we explore extremely flexible yet interpretable neural networks.

### Introduction and background

Deep learning approaches are often touted for having greater accuracy than other machine learning approaches, and this is often true.

### Input flexibility and the weaknesses of recurrent neural net architectures

### Abstract sequence encoding

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
