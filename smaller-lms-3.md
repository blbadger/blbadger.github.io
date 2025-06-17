### Language Mixers III: Optimization

Training large models takes a large amount of compute, but why this is the case is not immediately apparent as it does not necessarily follow from the model size alone. On this page we take a first-principles approach to understanding why this is the case through the lens of numerical optimization. We will explore the use of optimization methods that are theoretically much faster than gradient descent in the context of linear transformation optimization.

### Background

The process of training the deep learning models that are used for language and vision modeling may be thought of as a specialized type of unconstrained nonlinear optimization, one in which the function or model being optimized is itself composed of many linear transformations separated by various normalizations and nonlinear functions, where each linear transformation and linear part of the normalization (but typically none of the nonlinear functions).

If one were to take a number of mid-20th-century numerical analysts and show them the optimization procedures used today for large deep learning models, I think it would be safe to say that they would be fairly suprised at the use of methods that give virtually no guarantees of convergence, and that converge slowly when they do at all with respect to asymptotic order characteristics. This is because there have long been known to exists methods by which linear transformations may be solved explicitly, or for that matter methods that minimize nonlinear transformations with much faster convergence characteristics, that are completely absent from today's optimization methods.

This work will focus on the use of two methods for optimization that are in theory much faster than gradient descent, and detail why they are difficult to apply to even relatively simple language models. We then explore what exactly is necessary for convergence of large language models in the context of gradient descent-based optimization. 

### Introduction: Two classical optimization methods that converge quickly

For most machine learning tasks, we are given a system of equations (a 'model') and want to maximize the performance of this model on some given task, which is usually framed as minimizing a loss function on that task. If one is given a list of equations and asked to provide values that minimize the output of this equation, the approach one would want to take depends considerably on what type of equations are given. 

If the equations are linear and the loss function is a quadratic function like mean squared error, we can use some principles from calculus to solve the equation in one mathematical operation. It should be noted that this one operation will almost always require many computational iterations to complete, so it should not strictly speaking be thought of as only one step in a machine. Nevertheless, this can be viewed as perhaps the fastest optimization method possible.

If the equations are nonlinear, or the loss function is more complicated than a quadratic (and in particular if it is non-convex) then we must turn to other optimization methods that are typically iterative in nature. The methods commonly used for deep learning model optimization are based on gradient descent, in which many small steps are taken each in the direction of steepest decline in the loss function. If gradient descent-based optimization converges (meaning that it reaches a sufficiently small loss value) then the convergence occurs linearly: as we take a step of no more than the learning rate size for each iteration, we need at least $n$ steps for our initial model to reach the point of convergence. More precisely, 

$$
|| x^{(k+1)} - x_* || = O(|| x^{(k)} - x_* ||) \tag{eq1}
$$

This is somewhat of an oversimplification of today's methods, the most common of which being AdamW which typically one uses adaptive learning rates to estimate the momentum of a point in its loss trajectory, such that the accuracy of applying \ref{eq1} may be questioned. However, it can be shown that AdamW and other moment-estimating methods cannot converge any more quickly than pure gradient descent with well-chosen parameters (ref) such that \ref{eq1} is valid.

Happily there are other methods that converge much more quickly: one in particular is Newton's method. This name is somewhat confusing because it is applied to two different different optimization techniques, one that is essentially an improvement on gradient descent but requires the computation of Hessian matrices (which is usually computationally infeasible for large models) and one that requires only the Jacobian of the weight matrix, which in the case of functions $F: \Bbb R^m \to \Bbb R^1$ is equivalent to the gradient. This method is iterative but takes large steps, solving a linear equation for an intercept at each step. Newton's method converges quadratically provided some mild assumtions are made as to the continuity and nonsingularity of the system of equations,

$$
|| x^{(k+1)} - x_* || = O(|| x^{(k)} - x_* ||^2) \tag{eq2}
$$

This may not seem like much of an improvement on gradient descent, but it is actually an enormous improvement in terms of asymptotic characteristics.


### Newton's method






