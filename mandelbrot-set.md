## Julia and Mandelbrot sets with variations

The Julia set is the boundary of the sets of unbounded and bounded iterates of

$$
f_a(x) = x^2 + a 
\tag{1}
$$

where $a$ is fixed and $x_0$ varies about the complex plane $x + yi$.  This means that any number in the complex plane is in the Julia set if it borders another number $u$ such that $f^k_a(u) \to \infty \; \mathbf {as} \; k \to \infty$ as well as a number $c$ where $f^k_a(c) \not\to \infty \; \mathbf {as} \; k \to \infty$.

The Mandelbrot set $\mathscr M$ is the set of points $a$ in the complex plane for which the Julia sets are connected.  This happens to be the same set of points that iterations of the equation

$$
z = z^2 + a
\tag{2}
$$

do not diverge (go to positive or negative infinity) but instead are bounded upon many iterations at a starting value of $z = 0$.  Thus the Mandelbrot set is very similar to the Julia set but instead of fixing $a$ and ranging about $z$, the starting value of $z$ is fixed at 0 and the value of $a$ is ranged about the complex plane.  

The Mandelbrot set is a very rich fractal. Here is a small zoom, longer ones may be found in high quaility on Youtube

![disappearing mandelbrot]({{https://blbadger.github.io}}fractals/mandelbrot_zoom1.gif)

What happens if we add a small amount $b$ to $a$?  Then we have $x = x^2 + a + b$, and intuitively one can guess that the bounded set size will decrease as $b$ gets farther from the origin because there is less bounded area far from the origin in the Mandelbrot set. Let's look at many values of a real $b$, going from $b=0 \to b=1.3 \to b=0$:

![disappearing mandelbrot]({{https://blbadger.github.io}}fractals/mandelbrot_disappeared.gif)


How about a complex number? The set from $b = 0 \to b = 1 - i$ looks like

![disappearing complex mandelbrot]({{https://blbadger.github.io}}fractals/mandelbrot_complex_disappeared.gif)

