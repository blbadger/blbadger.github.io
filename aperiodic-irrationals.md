
## Periodicity and rationality

Here we establish an equivalence between irrational numbers and continuous aperiodic differential functions to examine the nature of aperiodicity.

As there is no universally-agreed upon notation for the set of irrational numbers, here we take the set of irrational numbers to be the real numbers that are not rational, $ \Bbb I = \Bbb R - \Bbb Q$, or equivalently

$$ \Bbb I = \{ x \; \vert \; x \in \Bbb R \land x \notin \Bbb Q\} $$

Rational numbers are expressible as fractions, whereas irrationals are not, and all real numbers are either rational or irrational.  As $\Bbb R $ is uncountably infinite but $\Bbb Q$ is countably infinite, $\Bbb I$ is uncountably infinite, meaning that the cardinality of the set of irrationals is much larger than the cardinality of rationals, or in other words nearly every possible number is irrational.

$$
card \; \Bbb I  >> card \; \Bbb Q
$$

Now let's consider continuous differential functions, which may be periodic or aperiodic. Aperiodicity means that values of the function $a(x)$ never return to that of previous iterations:

$$
a^n(x_0) \neq a^k(x_0) \; \mathbf {if} \; x, k \in \Bbb N \; \mathbf {and} \; k \neq x
$$

The set of all continuous functions with aperiodic outputs can be defined as $\Bbb A$

$$ 
\Bbb A = \{a(x)\}
$$

Conversely, a periodic differential function is one which does revisit a previous point

$$
p^n(x_0) = p^k(x_0) \; \mathbf {for} \; \mathbf {some} \; x, k \in \Bbb N \; \mathbf {given} \; k \neq x
$$

and the set of all continuous periodic functions can be denoted as
 
$$
\Bbb P = \{ p(x) \}
$$

The set of all continuous functions is equivalent in size to the set of all real numbers $\Bbb R$.  Equivalence can be expressed as $\sim$ and signifies that the sets are of equal size.  As functions may be periodic or aperiodic,

$$
\Bbb A + \Bbb P \sim \Bbb R
$$

We can define a periodic differential function based on its periodicity: in this case, all periodic functions with period 1 are defined as being the same function. Using this definition, there exists a one-to-one correspondance between periodic functions and the set of natural numbers $\Bbb N$ because periodic functions may have any finite period,

$$
\Bbb P = \{ \mathbf {period} \; 1, \mathbf {period} \; 2, \mathbf {period} \; 3... \} \\
\Bbb N = \{1, 2, 3... \} \\
\Bbb P \mapsto \Bbb N \\
\Bbb P \sim \Bbb N
$$

And as the set of natural numbers is equivalent (in size) to the set of rationals, by transitivity we have

$$
\Bbb P \sim \Bbb N \sim \Bbb Q \\
\Bbb P \sim \Bbb Q
$$

Recall that the set of all continuous functions (denoted here as $\Bbb F$ is equivalent to $\Bbb R$.  As functions may be periodic (quasiperiodic or asymptotically periodic functions are included) or aperiodic, and as the set of periodic functions is equivalent to the set of rational numbers,

$$
\Bbb A = \Bbb F - \Bbb P \sim \Bbb R - \Bbb Q = \Bbb I \\
\Bbb A \sim \Bbb I
$$

Thus the outputs of the set of continuous functions with aperiodic iterations, $\Bbb A$ is equivalent to the set of all irrational numbers $\Bbb I$ $\square$.

Let's see if the implications of this make sense.  In particular, can we also set up a one to one correspondance between aperiodic functions and natural numbers? This would be a contradiction if so, but it turns out we cannot: if we define functions based on their periodicity then consider that aperiodic functions have infinite periodicity, ie each has a period of $\infty$, which cannot map to the set of natural numbers, which are finite.

Also consider the process of writing down a number by adding one digit at a time to a decimal.  Irrational numbers have non-repeating decimal expansions, whereas rational numbers contain digits that eventually repeat.  The process decimal expansion is either periodic (defined as a repeating sequence rather than returning to an original value) or not, and aperiodic decimal expansion results in an irrational number whereas periodic decimal epansion gives a rational.


### Aperiodicity is equivalent to sensitivity to initial conditions

Say that points close together are stable if they stay close together in the arbitrary future, or are unstable if they diverge.  

$$
stable \Rightarrow | f(x) - f({x+\epsilon}) | \leq |f^n(x) - f^n((x + \epsilon)) \; \forall \; n \in \Bbb N|
$$

where $\epsilon$ is an arbitrarily small number and iterations of $f(x)$ are denoted $f^n(x)$. 

Now suppose that all points are unstable, such that all points that start close together move apart after a number of iterations. This is equivalent to saying that a system is sensitive to changes in initial values.  Now being that every starting initial value will eventually have different outputs than the value's neighbor (for any 'neighbor size'), each initial value has a unique trajectory.  As there is an uncountably infinite number of possible starting values on the interval $(0,1]$, 

$$
| \{ x \; : \; x \in (0, 1] \} | = |\Bbb R|
$$

there is an uncountably infinite number of possible trajectories if trajectories no two trajectories are the same (they diverge).  

As we have above established an equivalence between functions with periodic trajectories and rational numbers, 

$$
\Bbb B \sim \Bbb Q
$$

and as there are countably many rational numbers, there are countably many periodic trajectories. But as there are uncountably many trajectories if the system is unstable everywhere, trajectories cannot be periodic. Thus, instability (sensitivity) at all initial points cannot lead to periodic trajectories. As trajectories may be periodic or aperiodic, instability at initial values everywhere leads to aperiodic trajectories.

This same reasoning may be used to show that periodic systems must be insensitive to initial conditions, as there are countably many periodic outputs but uncountably many inputs, so some inputs must be equivalent.

### The set of chaotic (aperiodic) systems is equivalent to the set of unsolveable problems

Let's define solvable equations to be those that are computable in finite time for all inputs.  There exists and elegent proof for the idea that nearly every decision problem (that outputs one of two options) is not computable in finite time [here](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/lecture-23-computational-complexity/).  This proof establishes that nearly all decision (output 'yes' or 'no') problems are unsolveable in finite time, and can be tweaked to establish an equivalence between uncomputable (by a finite algorithm) problems and irrational numbers as follows:

Any program to compute a decision may be represented as a binary string of finite length, which is also a representation for any natural number.  

$$ 
program \approx binary \; string \approx x \in \Bbb N
$$

Now rational numbers are just two natural numbers divided by each other so any finite program can be represented by a rational number.

$$ 
x \in \Bbb Q \;if \; x = \frac{a}{b}, \; |\; a, b \in \Bbb N \\
program \approx binary \; string \approx x \in \Bbb Q
$$

The solution to any decision problem may be denoted as an infinite string of bits, 1 for 'yes' or 0 for 'no' for any number of infinite inputs, which corresponds to any real number.  As any program to compute a decicision is finite, and as this program may be represented by a rational number then only infinite strings of bits that are also representations of rationals may be computable. All other problems are unsolvable in finite time (with a finite program), and as any member of this set of problems is not rational but in the real numbers

$$
\{unsolvable \; problems\} \sim \Bbb R - \Bbb Q \sim \Bbb I
$$

Now as we have established an equivalence between irrational numbers and aperiodic (chaotic) systems above, by transitivity we can establish an equivalence between the set of unsolvable decision problems and the set of all outputs of chaotic systems or 

$$ 
\Bbb A \sim \Bbb I \equivalencesim \{unsolvable \; problems\}
$$

Thus chaotic systems are unsolvable in finite time.










