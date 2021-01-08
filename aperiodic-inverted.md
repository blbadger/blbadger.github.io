## Periodicity and reversibility

Consequences of non-invertability of aperiodic maps (logistic and Henon in particular) for dynamics.  

### The logistic map is non-invertible, hence dynamically irreversible

Suppose one were to track a dynamical system without knowing any details about how it worked.  

The logistic equation, which has been explored [here](https://blbadger.github.io/logistic-map.html) and [here](https://blbadger.github.io/logistic-boundary.html) is as follows:

$$
x_{n+1} = rx_n(1-x_n)
\tag{1}
$$

The logistic equation is a one-dimensional discrete dynamical map of very interesting behavior: periodicity for some values of $r$, aperiodicity for others, and for $0 < r < 4$, the interval $(0, 1)$ is mapped into itself. 

What does (1) look like in reverse, or in other words given any value $x_{n+1}$ which are the values of $x_n$ which when evaluated with (1) yield $x_{n+1}$?  Upon substituting $x_n$ for $x_{n+1}$, 

$$
x_n = rx_{n+1}(1-x_{n+1}) \\
0 = -rx_{n+1}^2 + rx_{n+1} - x_n \\
$$

The term $x_n$ can be treated as a constant (as it is constant for any given input value $x_n$), and therefore we can solve this expression for $x_{n+1}$ with the quadratic formula $ax^2 + bx + c$ where $a = -r$, $b = r$, and $c = -x_n$, to give

$$
x_{n+1} = \frac{r \pm \sqrt{r^2-4rx_n}}{2r}
$$

Now the first thing to note is that this dynamical equation is not strictly a function: it maps a single input $x_n$ to two outputs $x_{n+1}$ (one value for $+$ or $-$ taken in the numerator) for many values of $x_n \in (0, 1)$ whereas a function by definition has one output for any input that exists in the pre-image set.  



### Practical irreversibility in the aperiodic logistic map



### The Henon map is also non-invertible

The [Henon map](https://blbadger.github.io/henon-map.html) is a two dimensional discrete dynamical system defined as

$$
x_{n+1} = 1 - ax_n^2 + y_n \\
y_{n+1} = bx_n \\
\tag{2}
$$

Let's try the same method used above to reverse the $y_n$ component of (1) with respect to time:

$$
y_{n+1} = ax_{n+1}^2 + x_n - 1 \\
y_{n+1} = ay_n^2 / b^2 + x_n - 1
$$

and for $x_n$, 

$$
x_n = 1 - ax_{n+1}^2 + a(y_n/b)^2 + x_n -1 \\
x_{n+1}^2 = (y_n/b)^2 + 2x_n/a \\
x_{n+1} = \pm \sqrt{(y_n/b)^2 + 2x_n/a}
$$

Therefore the inverse of the Henon map is:

$$
x_{n+1} = \pm \sqrt{(y_n/b)^2 + 2x_n/a}
y_{n+1} = ay_n^2 / b^2 + x_n - 1
$$

Once again, the inverse is not a function as it may yield two values of $x_{n+1}$ for one input $x_n$, thus the Henon map like the logistic map is strictly speaking non-invertible as it too is 2-to-1 for certain values of $x_n, y_n$.  















