## Periodicity and reversibility

Suppose one were to observe movement over time, and wanted to describe the movement mathematically in order to be able to predict what happens in the future.  This is the goal of dynamical theory, and other pages on this site should convince one that precise knowledge of the future is much more difficult than it would seem even when a precise dynamical equation is known.  

What about if someone were curious about the past?  Given a system of dynamical equations describing how an object's movement occurs, can we find out where the object came from?  This question is addressed here for two relatively simple maps, the logistic and Henon.  

### The logistic map is non-invertible

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

Now the first thing to note is that this dynamical equation is not strictly a function: it maps a single input $x_n$ to two outputs $x_{n+1}$ (one value for $+$ or $-$ taken in the numerator) for many values of $x_n \in (0, 1)$ whereas a function by definition has one output for any input that exists in the pre-image set.  In other words the logistic map is non-invertible.

### Practical irreversibility in the aperiodic logistic map



### The Henon map is invertible but unstable

The [Henon map](https://blbadger.github.io/henon-map.html) is a two dimensional discrete dynamical system defined as

$$
x_{n+1} = 1 - ax_n^2 + y_n \\
y_{n+1} = bx_n \\
\tag{2}
$$

Let's try the same method used above to reverse the $y_n$ component of (2) with respect to time:

$$
y_{n+1} = ax_{n+1}^2 + x_n - 1 \\
y_{n+1} = a \frac{y_n^2}{b^2} + x_n - 1
$$

and for $x_n$, 

$$
y_{n} = bx_{n+1} \\
x_{n+1} = \pm \frac{y_n}{b}
$$

Therefore the inverse of the Henon map is:

$$
x_{n+1} = \frac{y_n}{b} \\
y_{n+1} = a \frac{y_n^2}{b^2} + x_n - 1
$$

This is a one-to-one map, meaning that (2) is invertible.  

Does this mean that, given some point $(x, y)$ in the plane we can determine the path it took to get there?  Let's try this with the reversal function above

```python
def reversed_henon_map(x, y, a, b):
	x_next = y / b
	y_next = a * (y / b)**2 + x - 1
	return x_next, y_next
```













