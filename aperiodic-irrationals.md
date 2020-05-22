## Periodicity and rationality

This section uses a one-to-one correspondance between irrational numbers and aperiodic systems to examine the nature of aperiodicity.

As there is no universally-agreed upon notation for the set of irrational numbers, here we take the set of irrational numbers to be the real numbers that are not rational, $ \Bbb I = \Bbb R - \Bbb Q$, or equivalently

$$ \Bbb I = \{ x \vert x \in \Bbb R \land x \notin \Bbb Q\} $$

Rational numbers are expressible as fractions, whereas irrationals are not.  As $\Bbb R $ is uncountably infinite but $\Bbb Q$ is countably infinite, $\Bbb I$ is uncountably infinite, meaning that nearly every possible number is irrational, or

$$
|\Bbb I | >> |\Bbb Q|
$$

WLOG, take an aperiodic iterative system such as the [logistic map](\logistic-map.md) at $r=4$ that outputs values between $(0, 1]$, and let's call this function $f_{a1}(x)$.  Any system with finite outputs can be transformed linearly such that it maps onto this interval.  Now the set of all iterative functions with aperiodic outputs on the interval $(0, 1]$ can be called 

$$ \mathbf A = \{ f_{a_i}(x) \} \; for \; all \; i \in \Bbb N \}
$$

For each function $f_{a_i}(x)$, we introduce a new function $g(x)$.  Now randomly assign each point on the interval $(0, 1]$ a digit from 0 to 9.  Now let's define a function $g(x)$ such that for each iteration of $f(x)$, the digit corresponding to the point that is the ouput to the end of $0.$.  As the number of iterations goes to infinity, a never-repeating sequence of digits appears: for example, $f(x) = 0.9385727050013 ...$ 

Every irrational number can be expressed as a sequence of digits after $0.$ that does not repeat, because for every division by a rational number there must be a remainder. Thus $g(f(x))$ is equivalent to a unique irrational number for any given starting value $x_0$.  As there are uncountably many initial values in the interval $(0, 1]$, there are uncountably many unique aperiodic trajectories for $f(x)$ and

$$\{g(f_{a_i}(x))\} = \Bbb I
$$

Thus the outputs of the set of functions with aperiodic iterations, $\{ f_a(x) \}$ when transformed using $g(x)$ is equal to the set of all irrational numbers $\Bbb I$. 


### An alternative proof that aperiodicity is equivalent to sensitivity to initial conditions

The correspondence between aperiodicity and irrational numbers may be used in a proof for Lorenz's observation that aperiodic ordinary differential equation systems are sensitive to changes in initial values:

Say that points close together are stable if they stay close together in the arbitraty future, or are unstable if they diverge.  Now suppose that all points are unstable, such that all points that start close together move apart after a number of iterations. This is equivalent to saying that a system is sensitive to changes in initial values.  Now being that every starting initial value will eventually have different outputs than the value's neighbor (for any 'neighbor size'), each initial value has a unique trajectory.  As there is an uncountably infinite number of possible starting values on the interval $(0,1]$, there is an uncountably infinite number of possible trajectories.  

As there is a one-to-one correspondence between periodic trajectories and rational numbers, and there are countably many rational numbers and thus countably many periodic trajectories if trajectories are periodic. But we have earlier seen that there must be uncountably many trajectories, thus trajectories must be aperiodic.  Therefore instability (sensitivity) at all initial points is equivalent to aperiodicity $\square$.  


### A proof for why chaotic systems are unsolveable

Let's define solveable equations to be those that are computable in finite time.  There exists and elegent proof for the idea that nearly every decision problem (that outputs one of two options) is not computable in finite time [here](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/lecture-23-computational-complexity/).  This proof establishes a one-to-one correspondence between unsolveable (by a finite algorithm) problems and irrational numbers.  Now as we have just established a one-to-one correspondence between irrational numbers and aperiodic (chaotic) systems, by transitivity we can establish a one-to-one correxpondence between the set of unsolveable decision problems and the set of all outputs of chaotic systems.  Thus chaotic systems are unsolveable in finite time.










