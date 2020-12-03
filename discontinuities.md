## Aperiodic phase space trajectories are discontinuous (in a certain dimension)

For this page, *aperiodic* signifies dynamical systems that are bounded (ie do not diverge to infinity) and lack periodicity.  Specifically, phase space portraits of ODEs are considered.  Unbounded dynamical systems may also be thought of as being aperiodic but are not considered.

### Continuous maps are defined on the rationals

This theorem has been established (see reference).

### Aperiodic trajectories in phase space cannot be defined in $\Bbb Q$

[Elsewhere](https://blbadger.github.io/aperiodic-irrationals.html) there is established an equivalence between the set of aperiodic trajectories and the set of irrational numbers.  This means that there are uncountably many aperiodic trajectories.  

Now define a phase space $S$ on $Q$ such that every coordinate is a rational number.  There are countably many rationals, so for any finite dimension $D$ there is a countable number of possible trajectories (as the trajectory repeats itself (ie becomes periodic) if it reaches any point it has seen previously).  With countably many inputs $S_0, S_1 ... $, so there are countably many functions that may exist in $S$.

But as there are uncountably many aperiodic trajectories, all but a meagre portion (ie an infinitely tiny amount) of these cannot be defined in $S$.  In the general case, therefore, aperiodic maps must be defined on the real (or irrational) numbers, not the rationals.

### Aperiodicity implies discontinuity

Thus continuous maps may be defined on the set of rational numbers $Q$, whereas the general aperiodic phase space maps cannot be defined on rationals.  Therefore the general aperiodic phase space map cannot be continuous.

### Why continuous maps can only be aperiodic and bounded in 3 or more dimensions

If continuous maps may form aperiodic attractors in three or more dimensions, why are they unable to do so in two or less?  A geometric argument is as follows: a three dimensional curve's intersection with a plane may be discontinuous.  A good example of this is found in the Lorenz attractor. 

![lorenz attractor]({{https://blbadger.github.io}}misc_images/lorenz_1.png)

A two dimensional curve's intersection with a line may be discontinuous as well. Why can't two dimensional continuous dynamical trajectories also be aperiodic?  

Smale introduced the horseshoe map to describe a common aperiodic attractor topology.  This model maps a square into itself such that the surface is stretched, folded and then re-stretched over and over like a baker's dough.  The horseshoe map is everywhere discontinuous as points arbitrarily close together to start are separated (the surface is stretched at each iteration and there are infinite iterations) such that there is a distance greater than some $e > 0$ between them.  The process of folding prevents divergence to infinity.  

The Poincare-Bendixon theorem states that no continuous, bounded dynamical system of two dimensions can form an aperiodic attractor but instead must be periodic or asymptotically periodic.  
