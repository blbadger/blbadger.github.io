Head: 
$$
D=2 \implies \forall f \; \exists n, k, n \neq k : f_c^n(x) = f_c^k(x)
$$

For this page, *aperiodic* signifies dynamical systems that are bounded (ie do not diverge to infinity) and lack periodicity.  Specifically, phase space portraits of ODEs are considered.  Unbounded dynamical systems may also be thought of as being aperiodic but are not considered.  On this page but not elsewhere, a 'map' is synonymous with a 'function'. 


### The Poincare-Bendixson Theorem: continuous phase space functions in two dimensions are periodic

Here 'dimensions' are taken to be plane dimensions.

If continuous maps may form aperiodic attractors in three or more dimensions, why are they unable to do so in two or less?  A geometric argument is as follows: a three dimensional curve's intersection with a line may be discontinuous and unrestricted.  For example, the Lorenz Attractor

![lorenz attractor]({{https://blbadger.github.io}}misc_images/lorenz_1.png)

A two dimensional curve's intersection with a line may be discontinuous as well. Why can't two dimensional continuous dynamical trajectories also be aperiodic?  The answer is that though the intersection may be discontinuous, it is restricted over time because trajectories are unique in phase space (ie lines cannot cross).

Here is a geometric argument: 

## Why discrete but not continuous maps in 2 dimensions exhibit fractal attractors

As seen with examples on [this page](https://blbadger.github.io/), movement from a continuous to a discrete-like map of continuous equations in phase space mapping $\Bbb R^2$ into $\Bbb R^2$ results in a shift from a point to a fractal attractor.  Note that these examples are only approximations of continuous maps, as in most cases one requires an infinite number of mathematical operations to produce a true continuous map of a nonlinear equation.  

There is a good reason for this shift: the Poincare-Bendixson theorem, which among other findings states that any attractor of a continuous map in two dimensions must be periodic: the attractor is either a point or a circuit line, corresponding to a zero- or one-dimensional attractor.  



### Discontinuous maps may be aperiodic in 1 or more dimension

In other pages on this site, it was made apparent that moving from a continuous to a discontinuous map is capable of transforming a periodic trajectory to an aperiodic one.  From the Poincare-Bendixson theorem, it is clear why this is the case: a restriction to any space as seen above is impossible if the trajectory may simply discontinuously pass through (jump through) previous trajectories.
