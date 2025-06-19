### Language Mixers III: Optimization

Training large models takes a large amount of compute, but why this is the case is not immediately apparent as it does not necessarily follow from the model size alone. On this page we take a first-principles approach to understanding why this is the case through the lens of numerical optimization. We will explore the use of optimization methods that are theoretically much faster than gradient descent in the context of linear transformation optimization.

### Background

The process of training the deep learning models that are used for language and vision modeling may be thought of as a specialized type of unconstrained nonlinear optimization, one in which the function or model being optimized is itself composed of many linear transformations separated by various normalizations and nonlinear functions, where each linear transformation and linear part of the normalization (but typically none of the nonlinear functions).

If one were to take a number of mid-20th-century numerical analysts and show them the optimization procedures used today for large deep learning models, I think it would be safe to say that they would be fairly suprised at the use of methods that give virtually no guarantees of convergence, and that converge slowly when they do at all with respect to asymptotic order characteristics. This is because there have long been known to exists methods by which linear transformations may be solved explicitly, or for that matter methods that minimize nonlinear transformations with much faster convergence characteristics, that are completely absent from today's optimization methods.

This work will focus on the use of two methods for optimization that are in theory much faster than gradient descent, and detail why they are difficult to apply to even relatively simple language models. We then explore what exactly is necessary for convergence of large language models in the context of gradient descent-based optimization. 

### Introduction: Classical optimization methods that converge quickly

For most machine learning tasks, we are given a system of equations (a 'model') and want to maximize the performance of this model on some given task, which is usually framed as minimizing a loss function on that task. If one is given a list of equations and asked to provide values that minimize the output of this equation, the approach one would want to take depends considerably on what type of equations are given. 

If the equations are linear and the loss function is a quadratic function like mean squared error, we can use some principles from calculus to solve the equation in one mathematical operation. It should be noted that this one operation will almost always require many computational iterations to complete, so it should not strictly speaking be thought of as only one step in a machine. Nevertheless, this can be viewed as perhaps the fastest optimization method possible.

If the equations are nonlinear, or the loss function is more complicated than a quadratic (and in particular if it is non-convex) then we must turn to other optimization methods that are typically iterative in nature. The methods commonly used for deep learning model optimization are based on gradient descent, in which many small steps are taken each in the direction of steepest decline in the loss function. If gradient descent-based optimization converges (meaning that it reaches a sufficiently small loss value) then the convergence occurs linearly: as we take a step of no more than the learning rate size for each iteration, we need at least $n$ steps for our initial model to reach the point of convergence. More precisely, 

$$
|| x_{(k+1)} - x^* || = O(|| x_{(k)} - x^* ||) 
\tag{1} \label{eq1}
$$

This is somewhat of an oversimplification of today's methods, the most common of which being AdamW which typically one uses adaptive learning rates to estimate the momentum of a point in its loss trajectory, such that the accuracy of applying \eqref{eq1} may be questioned. However, it can be shown that AdamW and other moment-estimating methods cannot converge any more quickly than pure gradient descent with well-chosen parameters (ref) such that \eqref{eq1} is valid.

Happily there are other methods that converge much more quickly: one in particular is Newton's method. This name is somewhat confusing because it is applied to two different related optimization techniques, one that is commonly applied to minimizing a loss function for nonlinear models and requires the computation of Hessian matrices (which is usually computationally infeasible for large models) and one that finds the roots of a function and requires only the Jacobian of the weight matrix, which in the case of functions $F: \Bbb R^m \to \Bbb R^1$ is equivalent to the gradient. This method is iterative but takes large steps, solving a linear equation for an intercept at each step. Newton's method converges quadratically provided some mild assumtions are made as to the continuity and nonsingularity of the system of equations,

$$
|| x_{(k+1)} - x^* || = O(|| x_{(k)} - x^* ||^2) 
\tag{2} \label{eq2}
$$

This may not seem like much of an improvement on gradient descent, but it is actually an enormous improvement in terms of asymptotic characteristics. Say you were given some target $x^*$ that required $n$ steps of gradient descent to reach from an initial value $x_0$. The same target would be expected to require only $\sqrt{n}$ steps for Newton's method, which is a substantial improvement as $n$ increases.

### Ordinary Least Squares via the Normal Equation

Suppose we were told that a linear model would be perfectly acceptable for language tasks and that we can choose to minimize any loss function we wanted. In this case perhaps the simplest conceptual approach is to choose a quadratic loss function like mean squared error and simply solve the resulting equation for the model parameters $\hat \beta$ given input $X$ and target output $y$, and for now we can avoid the details of exactly how this would be implemented. In the case of a $\beta$ being a square weight matrix of full rank, we can simply solve the model equation to get our desired weights (with zero loss):

$$
y = X\beta \implies X^{-1}y = \beta
$$


This is usually not possible as a weight matrix is in general non strictly invertible, such that there could be many or more commonly no exact solutions to the equation. In this case we have to decide on a loss function and find some value for our weights that minimizes this loss. In the common case in which there are no exact solutions for the equation, we can still solve for a desired weight value if the loss function is sufficiently simple. Mean squared error is happily simple enough to solve when applied to a linear regression, and there are a number of ways to approach this problem. 

One of the most straighforward ways of deriving this equation is to simply solve the system of equations present when we set the gradient to be zero,

$$
\Bbb L = || X \beta - y||_2^2
$$

which can be done via distribution and gradient application

$$
\nabla_\beta (1/m) (X\beta - y)^2 = 0 \implies \\
\nabla_\beta (\beta^TX^TX \beta - \beta^TX^Ty - y^TX\beta - y^Ty) = \\
2X^TX\beta - X^Ty - y^TX = 0
$$

Recalling that $y$ is a vector, we have $X^Ty = y^TX$ as both are the vector formed by the dot prods of columns of $X$ with y, so we have

$$
2X^TX\beta - 2X^Ty = 0 \implies \beta = (X^TX)^{-1}X^Ty
$$

This is unfortunately not the most numerically stable equation for larger matricies $X$: we can implement this equation directly using a tensor library like Pytorch as follows, and upon doing so we will not find that the parameters we obtain do not minimize our MSE loss function even for relatively small equations.

```python
beta_hat = (torch.inverse(X.T @ X) @ X.T) @ y
```

Happily, however, there is a more stable formulation utilizing the singular value decomposition components $U, D, V$,

$$
\theta_W = \lim_{\alpha \to 0^+} (X^TX + \alpha I)^{-1}X^T y \\
= X^+y = VD^+U^Ty
$$

where $X^+$ denotes the Roy-Moore Pseudo-inverse of $X$. This can be implemented simply in `torch` as follows:

```python
beta_hat = torch.pinverse(X) @ target
```

We will use a small linear language model to test the ability of the normal equation to minimize loss.

```
class LinearBlock(nn.Module):

    def __init__(self, dim, length):
        super().__init__()
        self.dim = dim
        self.length = length
        self.conv = nn.Conv1d(length, length, 1)

    def forward(self, x: torch.tensor):
        if x.dim() > 3:
            x = rearrange(x, 'b p t f -> (b p) t f')

        # for CLM training, apply lower triangular mask to convolution weights
        rearranged_shape = rearrange(self.conv.weight, 'f d p -> f (d p)').shape
        mask = torch.tril(torch.ones(rearranged_shape)).to(device)
        applied_mask = rearrange(self.conv.weight, 'f d p -> f (d p)') * mask
        self.conv.weight.data = rearrange(applied_mask, 'f (d p) -> f d p', p=1)

        residual = x
        x = self.conv(x) + residual
        return x
```

```
class LinearMixer(nn.Module):

    def __init__(self, n_vocab, dim, depth):
        super().__init__()
        self.wte = nn.Embedding(n_vocab, dim)
        self.mixerblocks = nn.ModuleList(
            [LinearBlock(
                dim = dim,
                length = tokenized_length
                )
            for i in range(depth)]
            ).to(device)
        self.lm_head = nn.Linear(dim, n_vocab, bias=False)
        self.mse = nn.MSELoss()

    def forward(self, input_ids, labels=None):
        x = input_ids
        x = x.to(device)
        x = self.wte(x)
        for block in self.mixerblocks:
            x = block(x)
        
        if labels is not None:
            x_prelim = x
            output = self.lm_head(x)
            labels = rearrange(labels, 'b p t -> b (p t)')
            output = rearrange(output, 'b t e -> b e t')
            shift_labels, shift_logits = labels, output
            shift_logits = output[..., :-1].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # convert labels to one hots, compute mse
            one_hots = torch.nn.functional.one_hot(shift_labels, num_classes=len(tokenizer)).transpose(1,2) 
            converted_labels = torch.tensor(one_hots, requires_grad=False, dtype=torch.double)
            loss = self.mse(shift_logits, converted_labels)
            return loss, output, x_prelim
        else:
            return x
```

```python
  @torch.no_grad()
  def normal_solve(model, train_data):
      train_batch = torch.stack(train_data[0:1], dim=0).to('cuda')
      loss, output, X = model(train_batch, labels=train_batch)
      target = torch.nn.functional.one_hot(train_batch, num_classes=len(tokenizer)).to(torch.double).squeeze(1)
      X = X[:, :-1, :].contiguous()
      target = target[:, 1:, :].contiguous()
      beta_hat = torch.pinverse(X) @ target
      model.lm_head.weight = torch.nn.Parameter(beta_hat.T)
      loss, output, X = model(train_batch, labels=train_batch) 
      return loss.item()
```

### Newton's method

As mentioned in the last section, there are two related versions of Newton's method that may be applied to the problem of minimizing an model's loss function given some data. It is helpful to first understand the differences between these methods and show how they are related before examining how they can be applied to deep learning models. At its heart, Newton's method is an iterative method that computes the roots (zeros) of an arbitrary equation $f(x)$: given an initial point $x_0$, we perform some number of iterations such that for each iteration at point $x_n$ we compute the next point $x_{n+1} via \eqref{eq3).

$$
x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} 
\tag{3} \label{eq3}
$$

The derivation of this formula is straightforward: given $f$ differentiable at point $x_n$ by definition $f'(x_n) = \delta y / \delta x$. We solve for the $\delta x$ such that $y=0$, meaning that $y=f(x_n)$ and 

$$
\frac{f(x_n)}{\delta x} = f'(x_n) \implies \Delta x = \frac{f(x_n}{f'(x_n)}
$$

and add this value's inverse (additive inverse) to our original point $x_n$ to obtain the next point.

$$
x_{n+1} = x_n - \Delta x = x_n - \frac{f(x_n)}{f'(x_n)}
$$

In the case where our input $x$ is a vector rather than a scalar and our function $f: \Bbb R^n \to \Bbb R^m$ is of many variables, we must compute the Jacobian $J$ of $f$, which is defined as 

$$
J_{ij}(x) = \frac{\delta f_i}{ \delta x_j}x
$$

$$
x_{n+1} = x_n - \Delta x = x_n - \frac{f(x_n)}{f'(x_n)}
$$

For \eqref{eq3} to actually find a root after many iterations, we need a few conditions to be satisfied (Lipschitz continuity and nonsingularity of $f'$ at $x_n$ for example) but most importantly there must actually be a root

$$
x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)} \label{eq4}
$$


















