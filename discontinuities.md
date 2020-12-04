## Continuity, aperiodicity, and Poincare-Bendixon

For this page, *aperiodic* signifies dynamical systems that are bounded (ie do not diverge to infinity) and lack periodicity.  Specifically, phase space portraits of ODEs are considered.  Unbounded dynamical systems may also be thought of as being aperiodic but are not considered.  On this page, a 'map' is synonymous with a 'function'.  Each heading is a theorem, with proof following (sometimes informal). 

### There are $2^{\Bbb Q} = \Bbb R$ continuous functions from $\Bbb Q \to \Bbb Q$

This theorem has been established and may be found in texts on real analysis, and one proof will suffice here.

First we define a 'unique' function to define a single trajectory in finite dimension $D$.  In other words, there is a one-to-one and onto mapping of a function to a trajectory.  The coordinates for this trajectory are defined by members of the set of rational numbers $\Bbb Q$ for any continuous function.  Now note that at any one point along a continuous trajectory, the next point is one of three options: it is slightly larger, slightly smaller, or the same. Graphically in two dimensions: 

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

### There are $2^{\Bbb R}$ discontinuous functions from $\Bbb Q \to \Bbb Q$

Discontinuous trajectories in $Q$ may do more than increase or decrease by $\delta$ or else stay the same: the next point may be any element of $\Bbb Q$.   The size of the set of all discontinuous functions is therefore the size of the set of all subsets of continuous functions. As we have established that the set of all continuous functions from $\Bbb Q \to \Bbb Q$ is equivalent to $\Bbb R$, 

$$
2^{\Bbb Q}^{\Bbb Q} = \Bbb R ^ {\Bbb R} = 2^{\Bbb R}
$$

### Discontinuous maps cannot be defined on the rationals

Functions are simply subsets of the Cartesian product of one set into another, meaning that a function mapping $\Bbb Q \to Bbb Q$ is a subset of 

$$
\Bbb Q^{\Bbb Q} = 2^{\Bbb Q}
$$

Thus there can be at most $\Bbb R$ functions mapping $\Bbb Q \to \Bbb Q$, but we have already seen that the size of the set of discontinuous functions is $2^{\Bbb R}$.  This means that discontinuous functions cannot be defined on the set of rational numbers $Q$ for any finite dimension $D$.

### Aperiodic trajectories in phase space cannot be defined on the rationals

tbc

### Aperiodic trajectories are discontinuous*

tbc

### The Poincare-Bendixon Theorem: continuous phase space functions in two dimensions are periodic

If continuous maps may form aperiodic attractors in three or more dimensions, why are they unable to do so in two or less?  A geometric argument is as follows: a three dimensional curve's intersection with a plane may be discontinuous.  A good example of this is found in the Lorenz attractor. 

![lorenz attractor]({{https://blbadger.github.io}}misc_images/lorenz_1.png)

A two dimensional curve's intersection with a line may be discontinuous as well. Why can't two dimensional continuous dynamical trajectories also be aperiodic?  

Smale introduced the horseshoe map to describe a common aperiodic attractor topology.  This model maps a square into itself such that the surface is stretched, folded and then re-stretched over and over like a baker's dough.  The horseshoe map is everywhere discontinuous as points arbitrarily close together to start are separated (the surface is stretched at each iteration and there are infinite iterations) such that there is a distance greater than some $e > 0$ between them.  The process of folding prevents divergence to infinity.  

### Discontinuous maps may be aperiodic in 1 or more dimension

tdc


