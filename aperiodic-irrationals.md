## Periodicity and rationality

This section uses a one-to-one correspondance between irrational numbers and aperiodic systems to examine the nature of aperiodicity.

As there is no universally-agreed upon notation for the set of irrational numbers, here we take the set of irrational numbers to be the real numbers that are not rational, $ \Bbb I = \Bbb R - \Bbb Q$, or equivalently

$$ \Bbb I = \{ x \vert x \in \Bbb R \land x \notin \Bbb Q\} $$

Rational numbers are expressible as fractions, whereas irrationals are not, and all real numbers are either rational or irrational.  As $\Bbb R $ is uncountably infinite but $\Bbb Q$ is countably infinite, $\Bbb I$ is uncountably infinite, meaning that the cardinality of the set of irrationals is much larger than the cardinality of rationals, or in other words nearly every possible number is irrational.

$$
|\Bbb I | >> |\Bbb Q|
$$

Take an aperiodic function system such as the [logistic map](\logistic-map.md) at $r=4$ that outputs values between $(0, 1]$, and let's call this function $f_{a_1}(x)$.  Another aperiodic function mapping to the same interval may be calles $f_{a_2}(x)$.  Any system with finite outputs can be transformed linearly such that it maps onto this interval.  Now the set of all iterative functions with aperiodic outputs on the interval $(0, 1]$ can be called 

$$ 
\Bbb A = \{ f_{a_i}(x) \} \; for \; all \; i \in \Bbb N 
$$

For each function $f \in \Bbb A$, we introduce a new function $g(y)$.  Each point on the interval $(0, 1]$ is randomly assigned a digit from 0 to 9.  $g(y)$ takes the outputs of infinite iterations of $f(x)$ and maps them to a sequence of digits after $0.$, ie $0.738510279...$. 

Every irrational number can be expressed as a sequence of digits after $0.$ that does not repeat.  Thus $g(f_{a_i}(x))$ is equivalent to a unique irrational number for any given starting value $x_0$.  If we take the set of all outputs of $ \Bbb A $ 

$$\{g(f_{a_i}(x))\} \; for\;all\;i \in \Bbb N = \Bbb I \\
\Bbb A \approx \Bbb I
$$

Thus the outputs of the set of functions with aperiodic iterations, $\{ f_a(x) \}$ when transformed using $g(y)$ is equal to the set of all irrational numbers $\Bbb I$.


Conversely, take the set of all functions that map onto the interval $(0, 1]$, 

$$
\Bbb B = \{ h_{a_i}(x) \} \; for \; all \; i \in \Bbb N
$$

Map the outputs of any element of $\Bbb B$ to a sequence of digits after $0.$ with $g(y)$ as above.  For any starting value $x_0$, the output of a single function mapped to digits, $g(h_{a_i}(x))$, is only guaranteed to yeild a periodic digit sequence for all iterations if $h_{a_i)(x)$ is periodic.  Thus 

$$
\{g(h_{a_i}(x))\} \;for\;all\;i \in \Bbb N = \Bbb Q \\
\Bbb B \approx \Bbb Q
$$

or in other words, periodic functions correspond to the set of all rational numbers when transformed by $g(y)$. 


### Aperiodicity is equivalent to sensitivity to initial conditions

Say that points close together are stable if they stay close together in the arbitraty future, or are unstable if they diverge.  



Now suppose that all points are unstable, such that all points that start close together move apart after a number of iterations. This is equivalent to saying that a system is sensitive to changes in initial values.  Now being that every starting initial value will eventually have different outputs than the value's neighbor (for any 'neighbor size'), each initial value has a unique trajectory.  As there is an uncountably infinite number of possible starting values on the interval $(0,1]$, there is an uncountably infinite number of possible trajectories.  

As we have established a correspondence between periodic trajectories and rational numbers, there are countably many rational numbers and thus countably many periodic trajectories if trajectories are periodic. But we have earlier seen that there must be uncountably many trajectories, thus trajectories must be aperiodic.  By contradiction, instability (sensitivity) at all initial points is equivalent to aperiodicity $\square$.  


### Chaotic systems are unsolveable

Let's define solveable equations to be those that are computable in finite time for all inputs.  There exists and elegent proof for the idea that nearly every decision problem (that outputs one of two options) is not computable in finite time [here](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/lecture-23-computational-complexity/).  This proof establishes that nearly all decision (output 'yes' or 'no') problems are unsolveable in finite time, and can be tweaked to establish an equivalence between uncomputable (by a finite algorithm) problems and irrational numbers as follows:

Any program to compute a decision may be represented as a binary string of finite length, which is also a representation for any natural number.  

$$ program \approx binary string \approx x \in \Bbb N
$$

Now rational numbers are just two natural numbers divided by each other so any finite program can be represented by a rational number.

$$ x \in \Bbb Q \;if x= \frac{a}{b}, \; |\; a, b \in \Bbb N \\
program \approx binary string \approx x \in \Bbb Q
$$

The solution to any decision problem may be denoted as an infinite string of bits, 1 for 'yes' or 0 for 'no' for any number of infinite inputs, which corresponds to any real number.  As any program to compute a decicision is finite, and as this program may be represented by a rational number then only infinite strings of bits that are also representations of rationals may be computable. All other problems are unsolveable in finite time (with a finite program), and as any member of this set of problems is not rational but in the real numbers

$$
{unsolveable problems} = \Bbb R - \Bbb Q = \Bbb I
$$

Now as we have established an approximate equivalence between irrational numbers and aperiodic (chaotic) systems above, by transitivity we can establish an approximate equivalence between the set of unsolveable decision problems and the set of all outputs of chaotic systems or 

$$ \Bbb A \approx \Bbb I = {unsolveable problems}
$$

Thus chaotic systems are unsolveable in finite time $\square$










