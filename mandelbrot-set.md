## Mandelbrot set with variations

The Mandelbrot set $\mathscr M$ is the set of points $a$ in the complex plane for which the Julia sets are connected.  This happens to be the same set of points that iterations of the equation

$$
z = z^2 + a
\tag{2}
$$

do not diverge (go to positive or negative infinity) but instead are bounded upon many iterations at a starting value of $z = 0$.  Thus the Mandelbrot set is very similar to the Julia set but instead of fixing $a$ and ranging about $z$, the starting value of $z$ is fixed at 0 and the value of $a$ is ranged about the complex plane.  

The Mandelbrot set is a very rich fractal. Here is a zoom on the point - 0.74797 + 0.072500001i.  

![disappearing mandelbrot]({{https://blbadger.github.io}}fractals/mandelbrot_zoom1.gif)

And here is the same point, increasing scale to a factor o $2^{42}$ (over four trillion)

<iframe width="560" height="315" src="https://www.youtube.com/embed/0qrordbf7WE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

What happens if we change the exponent of (2) such that $z^1 \to z^4$ ?  At $z^1$, the equation is linear and a circular region about the origin remains bounded.  But as the system becomes nonlinear, intricate shapes appear.

![extended mandelbrot]({{https://blbadger.github.io}}fractals/mandelbrot_slow.gif)


What happens if we add a small amount $b$ to $a$?  Then we have $x = x^2 + a + b$, and intuitively one can guess that the bounded set size will decrease as $b$ gets farther from the origin because there is less bounded area far from the origin in the Mandelbrot set. Let's look at many values of a real $b$, going from $b=0 \to b=1.3 \to b=0$:

![disappearing mandelbrot]({{https://blbadger.github.io}}fractals/mandelbrot_disappeared.gif)


How about a complex number? The set from $b = 0 \to b = 1 - i$ looks like

![disappearing complex mandelbrot]({{https://blbadger.github.io}}fractals/mandelbrot_complex_disappeared.gif)

