

## The boundary of the logistic equation

Why aren't values for r>4 shown in any of the graphs above? This is because these values head towards infinity: they are unbounded.  Which values are bounded? 

Complex plane equations (3). 

Let's try fixing $r$ in place, say at $3$ and seeing what happens to different starting points in the complex plane after iterations of (3).  

![complex map]({{https://blbadger.github.io}}/logistic_map/logistic_boundary_3_fixed_r.png)

The result is nearly identical to a [Julia set](/julia-sets.md)) for $a = -0.75 + 0i$, shown below.

![julia map]({{https://blbadger.github.io}}/logistic_map/julia_-0.75.png)

Just for fun, let's zoom in on the origin.  For most decreases in scale, more iterations are required in order to determine if close-together coordinates will diverge towards infinity or else remain bounded.  But this is not the case for a zoom towards the origin: no more iterations are required for constant resolution even when the scale has increased by a factor of $2^{20}$.

![complex map]({{https://blbadger.github.io}}/logistic_map/logistic_bound_fixed_r.gif)

Moving from $r=2 \to r=4$, 

![complex map]({{https://blbadger.github.io}}/logistic_map/logistic_boundary_fixed_r.gif)

What happens if we instead fix the starting point and allow $r$ to range about the complex plane? For the starting point $x_0, yi_0 = 0.5, 0$, we get 

![complex map]({{https://blbadger.github.io}}/logistic_map/logistic_bound_0.5.png)

And from $x_0 = 0 \to x_0 = 2$, 

![complex map]({{https://blbadger.github.io}}/logistic_map/logistic_boundary_fixed_start.gif)

At $(x_0, yi_0) = (0.5, 0) = 0.5 + 0i$ figure resembles a double-sided [Mandelbrot set](/mandelbrot-set.md).  When we zoom in, we can find many little mandelbrot set shapes (in reversed x-orientation).  This intuitively makes sense: the Mandelbrot set is what we get when we iterate 

$$
z_{next} = z^2 + c
$$

in the complex plane and find which positions ($c$) head towards infinity and which do not for a given starting point $z_0$.  This is analagous to fixing the starting point for the logistic equation (at $(x_0, y_0) = (0.5, 0i)$ in this case) and then looking at which $r$ values cause iterations of the logistic equation in the complex plane

$$
z_{next} = rz(1-z) = rz - rz^2
$$

For example, increasing the scale at the point $(3.58355 + 0i)$ yeilds

![complex map]({{https://blbadger.github.io}}/logistic_map/logistic_bound_zoom.gif)
