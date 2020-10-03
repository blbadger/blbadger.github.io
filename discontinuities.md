## Discontinuity is necessary for aperiodicity

For this page, *aperiodic* signifies dynamical systems that are bounded (ie do not diverge to infinity) and lack periodicity.  Specifically, phase space portraits of ODEs are considered.  Unbounded dynamical systems may also be thought of as being aperiodic but are not considered.

### Continuous maps are defined on the rationals

### Continuous maps can only be aperiodic and bounded in 3 or more dimensions

The Poincare-Bendixon theorem states that no continuous, bounded dynamical system of two dimensions can form an aperiodic attractor but instead must be periodic or asymptotically periodic.  

### Aperiodicity implies discontinuity

If continuous maps may form aperiodic attractors in three or more dimensions, why are they unable to do so in two or less?  A geometric argument is as follows: a three dimensional curve's intersection with a plane may be discontinuous.  A good example of this is found in the Lorenz attractor. 

![lorenz attractor]({{https://blbadger.github.io}}misc_images/lorenz_1.png)

A two dimensional curve's intersection with a line may be discontinuous as well. Why can't two dimensional continuous dynamical trajectories also be aperiodic?  

Smale introduced the horseshoe map to describe a common aperiodic attractor topology.  This model maps a square into itself such that the surface is stretched, folded and then re-stretched over and over like a baker's dough.  The horseshoe map is everywhere discontinuous as points arbitrarily close together to start are separated (the surface is stretched at each iteration and there are infinite iterations) such that there is a distance greater than some $e > 0$ between them.  The process of folding prevents divergence to infinity.  


