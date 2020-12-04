## Continuity, aperiodicity, and Poincare-Bendixon

For this page, *aperiodic* signifies dynamical systems that are bounded (ie do not diverge to infinity) and lack periodicity.  Specifically, phase space portraits of ODEs are considered.  Unbounded dynamical systems may also be thought of as being aperiodic but are not considered.  On this page, a 'map' is synonymous with a 'function'.  Each heading is a theorem, with proof following (sometimes informal). 

### There are $2^{\Bbb Q} = \Bbb R$ continuous functions

This theorem has been established and may be found in texts on real analysis, and one proof will suffice here.

First we define a 'unique' function to define a single trajectory in finite dimension $D$.  In other words, there is a one-to-one and onto mapping of a function to a trajectory.  The coordinates for this trajectory are defined by members of the set of rational numbers $\Bbb Q$.  Now note that at any one point along a continuous trajectory, the next point is one of three options: it is slightly larger, slightly smaller, or the same. Graphically in two dimensions: 

![continuous function next value]({{https://blbadger.github.io}}misc_images/continuous_function_next.png)

and precisely, $f$ may increase by $\delta$, decrease by $\delta$, or stay the same where $delta$ is defined as 

$$
\lvert x_1 - x_2 \rvert < \epsilon \implies \lvert f(x_1) - f(x_2) \rvert < \delta
$$

for any arbitrarily small value $\epsilon$. 

Thus for $\Bbb Q$ there are three options, meaning that the set of continuous functions is equivalent to the set of all sets of $\Bbb Q$ into ${0, 1, 2}$

$$
{f} \sim 3^{\Bbb Q} \sim \Bbb R
$$

or the set of all continuous functions defined on $\Bbb Q$ is equivalent to the set of real numbers. 

### Discontinuous maps cannot be defined on the rationals

Discontinuous trajectories in $Q$ may do more than increase or decrease by $\delta$ or else stay the same: the next point may be any element of $\Bbb Q$.   The size of the set of all discontinuous functions is 

$$
2^{\Bbb Q}^{\Bbb Q} = \Bbb R ^ {\Bbb R} = 2^{\Bbb R}
$$


### Aperiodic trajectories in phase space cannot be defined in $\Bbb Q$

Above and [Elsewhere](https://blbadger.github.io/aperiodic-irrationals.html) there is established an equivalence between the set of aperiodic trajectories and the set of irrational numbers (or the set of real numbers), which means that there are uncountably many aperiodic trajectories.  

Now define a phase space $S$ on $Q$ such that every coordinate is a rational number.  There are countably many rationals, so for any finite dimension $D$ there is a countable number of possible trajectories (as the trajectory repeats itself (ie becomes periodic) if it reaches any point it has seen previously).  With countably many inputs $S_0, S_1 ... $, so there are countably many functions that may exist in $S$.

But as there are uncountably many aperiodic trajectories, all but a meagre portion (ie an infinitely tiny amount) of these cannot be defined in $S$.  In the general case, therefore, aperiodic maps must be defined on the real (irrational) numbers, not the rationals.

This only applies to maps in phase space, where revisiting any previous point necessarily entails periodicity.  Functions can lead to aperiodic behavior in $\Bbb N$ or other countable sets if revisiting a previous point does not lead to

### In general, aperiodic maps are discontinuous

The Poincare-Bendixon theorem states that no continuous, bounded dynamical system of two dimensions can form an aperiodic attractor but instead must be periodic or asymptotically periodic.  

As above it is found that the set of all continuous maps may be defined on the set of rational numbers $Q$, whereas the set of aperiodic maps cannot be defined on rationals.  As there are infinitely many more aperiodic functions than continuous functions, nearly all but a negligable portion of aperiodic functions are discontinuous.

### Why continuous maps can be aperiodic and bounded in 3 or more dimensions

If continuous maps may form aperiodic attractors in three or more dimensions, why are they unable to do so in two or less?  A geometric argument is as follows: a three dimensional curve's intersection with a plane may be discontinuous.  A good example of this is found in the Lorenz attractor. 

![lorenz attractor]({{https://blbadger.github.io}}misc_images/lorenz_1.png)

A two dimensional curve's intersection with a line may be discontinuous as well. Why can't two dimensional continuous dynamical trajectories also be aperiodic?  

Smale introduced the horseshoe map to describe a common aperiodic attractor topology.  This model maps a square into itself such that the surface is stretched, folded and then re-stretched over and over like a baker's dough.  The horseshoe map is everywhere discontinuous as points arbitrarily close together to start are separated (the surface is stretched at each iteration and there are infinite iterations) such that there is a distance greater than some $e > 0$ between them.  The process of folding prevents divergence to infinity.  

