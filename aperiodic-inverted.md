## Periodicity and reversibility

Consequences of non-invertability of aperiodic maps (logistic and Henon in particular) for dynamics.  

### The logistic map is non-invertible, hence dynamically irreversible

Suppose one were to track a dynamical system without knowing any details about how it worked.

The logistic map, which has been explored [here](https://blbadger.github.io/logistic-map.html) and [here](https://blbadger.github.io/logistic-boundary.html), is a one-dimensional discrete dynamical system capable of very interesting behavior. 

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



