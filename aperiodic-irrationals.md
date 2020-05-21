## Aperiodic systems and irrational numbers

This section uses a one-to-one correspondance between irrational numbers and aperiodic systems to examine the nature of aperiodicity.

As there is no universally-agreed upon notation for the set of irrational numbers, here we take the set of irrational numbers to be the real numbers that are not rational, $ \Bbb I = \BbbR - \Bbb Q$, or equivalently

$$ \Bbb I = \{ x \vert x \in \Bbb R \land x \notin \Bbb Q\} $$

Rational numbers are expressible as fractions, whereas irrationals are not.  As $\Bbb R $ is uncountably infinite but $\Bbb Q$ is countably infinite, $\Bbb I$ is uncountably infinite, meaning that nearly every possible number is irrational. 

Take an aperiodic iterative system such as the [logistic map](\logistic-map.md) at $r=4$ that outputs values between $(0, 1]$.  Any system with finite outputs can be transformed linearly such that it maps onto this interval.  Now randomly assign each point on the interval $(0, 1]$ a digit from 0 to 9.  For each iteration of the aperiodic system, add the digit corresponding to the point that is the ouput to the end of $0.$.  As the number of iterations goes to infinity, the 

### An alternative proof that aperiodicity is equivalent to sensitivity to initial conditions

The equivalence between aperiodicity and irrational numbers may be used in a proof for Lorenz's observation that aperiodic ordinary differential equation systems are sensitive to changes in initial values:

Say that points close together are stable if they stay close together in the arbitraty future, or are unstable if they diverge.  Now suppose that all points are unstable, such that all points that start close together move apart after a number of iterations. This is equivalent to saying that a system is sensitive to changes in initial values.  Now being that every starting initial value will eventually have different outputs than the value's neighbor (for any 'neighbor size'), each initial value has a unique trajectory.  As there is an uncountably infinite number of possible starting values on the interval $(0,1]$, there is an uncountably infinite number of possible trajectories.  We can establish a one-to-one correspondence between periodic trajectories and rational numbers, and there are countably many rational numbers and thus countably many periodic trajectories if trajectories are periodic. But we have earlier seen that there must be uncountably many trajectories, thus trajectories must be aperiodic.  Therefore instability (sensitivity) at all initial points is equivalent to aperiodicity $\square$.  

### A proof for why chaotic systems are unsolveable








