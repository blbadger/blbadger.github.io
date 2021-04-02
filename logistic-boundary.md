## The boundary of the logistic map

### The logistic map is a conjugate of a Julia set

The logistic equation

$$
x_{n+1} = rx_n (1 - x_n) \\
\tag{1}
$$

is a model for growth rate that displays many features of nonlinear dynamics in a nice one-dimensional form. For a summary on some of these interesting properties, see [here](/logistic-map.md).

Why aren't values for r>4 shown in any of the graphs for the logistic equation? This is because these values head towards infinity: they are unbounded.  Which values are bounded? This question is not too difficult to answer: for r values of 0 to 4, initial populations anywhere in the range $[0, 1]$ stay in this range.  

What if we iterate (1) but instead of $x$ existing on the real line, we allow it to traverse the complex plane? Then we have

$$
z_{n+1} = rz_n (1 - z_n) \\
\tag{2}
$$

Where $z_n$ (and $z_{n+a}$ for any $a$ and $r$ are points in the complex plane:

$$
z_n = x_0 + y_0i \\
r = x_1 + y_1i
$$

Now which points remain bounded and which head towards infinity?  Let's try fixing $r$ in place, say at $3$ and seeing what happens to different starting points in the complex plane after iterations of (3).  

![complex map]({{https://blbadger.github.io}}/logistic_map/logistic_boundary_3_fixed_r.png)

The result looks very similar to a [Julia set](/julia-sets.md), is defined as the set of points bordering initial points whose subsequent iterations of 

$$
z_{n+1} = z_n^2 + a
\tag{3}
$$

that diverge (head to infinity) and points whose subsequent interations do not diverge.

For $a = -0.75 + 0i$, this set is shown below (with an inverted color scheme to that used for the logistic map for clarity)

![julia map]({{https://blbadger.github.io}}/logistic_map/julia_-0.75.png)

These maps look extremely similar, so could they actually be the same?  They are indeed!  The logistic map (1) and the quadratic map (3) which forms the basis of the Julia sets are conjugates of one another: they contain identical topological properties for a certain $a$ value, or in other words transforming from one map to another is a homeomorphism.  

This can be shown as follows: a linear transformation on any variable is the same as a (combination of) stretching, rotation, translation, or dilation.  Each of these possibilities does not affect the underlying topology of the transformed space, which one can think of as being true because the space is not broken apart in any way.  Therefore linear transformations do not affect 

This being the case, the logistic map (1) may be transformed into the quadratic map (3) with the linear transformation $y_n = ax_n + b$ by choosing certain values of $a, \; b$ as follows:

$$
y_{n+1} = y_n^2 + c, \; y_n = ax_n+b \\
x_{n+1} = (ax_n+b)^2 + c = a^2x_n^2 + 2abx_n + b^2 + c \\
b = a/2 \implies x_{n+1} = a^2(1-x_n) + (a/2)^2 + c \\
c=-(a/2)^2 \implies x_{n+1} = a^2x_n(1-x_n) \\
a^2 = r \implies x_{n+1} = rx_n(1-x_n)
$$

Therefore for $a=\sqrt{r},\; b=\sqrt{r}/2, \; c=-r/4$ the quadratic map is by a homeomorphism (ie a linear transformation) equivalent to the logistic map, and thus the two are topologically equivalent.  Now the necessity of the seemingly arbitrary value of $a=-0.75$ for the quadratic map is clear: $r=3$ was specified for the logistic map, and by our homeomorphism then $c=-r/4 = -3/4$.  

All this is to say that for any $r$ value, the logistic map is equivalent to a Julia set where $c=-r/4$. Just for fun, let's zoom in on the origin of the set displayed above.  An aside: for most decreases in scale, more iterations are required in order to determine if close-together coordinates will diverge towards infinity or else remain bounded.  But this is not the case for a zoom towards the origin: no more iterations are required for constant resolution even when the scale has increased by a factor of $2^{20}$.

![complex map]({{https://blbadger.github.io}}/logistic_map/logistic_bound_fixed_r.gif)

Moving from $r=2 \to r=4$, 

![complex map]({{https://blbadger.github.io}}/logistic_map/logistic_boundary_fixed_r.gif)

### The logistic map and the Mandelbrot set

What happens if we instead fix the starting point and allow $r$ to range about the complex plane? For the starting point $x_0, yi_0 = 0.5, 0$, we get 

![complex map]({{https://blbadger.github.io}}/logistic_map/logistic_bound_0.5.png)

And from $x_0 = 0 \to x_0 = 2$, 

![complex map]({{https://blbadger.github.io}}/logistic_map/logistic_boundary_fixed_start.gif)

At $(x_0, yi_0) = (0.5, 0) = 0.5 + 0i$ figure resembles a double-sided [Mandelbrot set](/mandelbrot-set.md).  When we zoom in, we can find many little mandelbrot set shapes (in reversed x-orientation).  This intuitively makes sense: the Mandelbrot set is what we get when we iterate 

$$
z_{next} = z^2 + c
$$

in the complex plane and find which positions ($c$) head towards infinity and which do not for a given starting point $z_0$.  This is analagous to fixing the starting point for the logistic equation (at $(x_0, y_0) = (0.5, 0i)$ in this case) and then looking at which $r$ values cause future iterations of 

$$
z_{next} = rz(1-z) = (-z^2 + z)r
$$

to head towards infinity. To illustrate, increasing the scale at the point 

$$
(3.58355 + 0i)
$$

yields

![complex map]({{https://blbadger.github.io}}/logistic_map/logistic_bound_zoom.gif)

Note the identical objects found in a Mandelbrot set zoom on the point 

$$
(-1.633 + 0i)
$$

Another aside: the previous and following zooms were made using a different method than elaborate on previously.  Rather than modifying the ogrid directly, `a_array` and `z_array` are instead positioned (using addition and subtraction of real and imaginary amounts) and then scaled over time as follows:

```python
def mandelbrot_set(h_range, w_range, max_iterations, t):
	y, x = np.ogrid[1.4: -1.4: h_range*1j, -1.8: 1:w_range*1j] # note that the ogrid does not scale

	a_array = x/(2**(t/15)) - 1.633 + y*1j / (2**(t/15)) # the array scales instead
	z_array = np.zeros(a_array.shape)
	iterations_till_divergence = max_iterations + np.zeros(a_array.shape)
  
  ...
```

This method requires far fewer iterations at a given scale for an arbitrary resolution relative to the method of scaling the ogrid directly, although more iterations are required for constant resolution as the scale decreases.  The fewer iterations is presumably due to decreased round-off error: centering the array and zooming in on the origin leads to approximately constant round-off error, whereas zooming in on a point far from the origin leads to significant error that requires more iterations to resolve.  I am not completely certain why a constant number of iterations are not sufficient for constant resolution using this method, however. 

![mandelbrot map]({{https://blbadger.github.io}}/logistic_map/mandelbrot_zoom_frame.gif)
