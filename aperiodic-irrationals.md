for index:

### [Aperiodic maps and irrational numbers](/aperiodic-irrationals.md)

$$ 
\Bbb R - \Bbb Q \sim \{f(x) : f^n(x(0)) \neq f^k(x(0))\} \\
\text{given} \; n, k \in \Bbb N \; \text{and} \; k \neq n
$$


## Periodicity and rationality

Here we establish a one to one correspondance between irrational numbers and aperiodic equations to examine the nature of aperiodicity.

As there is no universally-agreed upon notation for the set of irrational numbers, here we take the set of irrational numbers to be the real numbers that are not rational, $ \Bbb I = \Bbb R - \Bbb Q$, or equivalently

$$ \Bbb I = \{ x \; \vert \; x \in \Bbb R \land x \notin \Bbb Q\} $$

Rational numbers are expressible as fractions, whereas irrationals are not, and all real numbers are either rational or irrational.  As $\Bbb R $ is uncountably infinite but $\Bbb Q$ is countably infinite, $\Bbb I$ is uncountably infinite, meaning that the cardinality of the set of irrationals is much larger than the cardinality of rationals, or in other words nearly every possible number is irrational.

$$
card \Bbb I  >> card \Bbb Q
$$

Take an aperiodic iterated function $f: \Bbb R \to \Bbb R$ such as the [logistic map](\logistic-map.md) at $r=4$ that outputs values between $(0, 1]$, and let's call this function $a(x)$.  Aperiodicity for iterated functions means that values never return to previous iterations:

$$
a^n(x_0) \neq a^k(x_0) \; \mathbf {if} \; x, k \in \Bbb N \; \mathbf {and} \; k \neq x
$$

Another aperiodic function mapping to the same interval may be called $a_2(x)$ if at least one of its iterations is different than that obtained for $a(x)$.   
Now the set of all iterative functions with aperiodic outputs, $A$ on the interval $(0, 1]$ can be called 

$$ 
\Bbb A = \{a(x)\}
$$

For all functions $a \in \Bbb A$ we introduce a new function $f(x)$, which maps the output of this function onto the digits 0 through 9 simply by recording the first digit of the decimal expansion of $a^n(x)$, such that $a^1(x) = 0.243859... \to 2$, and repeated iterations are placed in decimel point $0.1252403...$.

Every irrational number can be expressed as a sequence of digits after $0.$ that does not repeat.  $f(x)$ maps non-repeating iterations to digits

Is this a one-to-one mapping, ie can any irrational number in $(0, 1]$ be expressed as $f(a(x))$ and does $f(a(x))$ map to a unique irrational number in $(0, 1]$? Consider the first question. Indeed any irrational number in $(0, 1]$ can be expressed with $f(a(x))$ as there is no limit to which digit can follow any other digit in our mapping.  For the second question, consider this: will $f(x)$ ever repeat for all iterations of an aperiodic function? 

Sets that have one-to-one mappings between them are considered equivalent, meaning that they are bound by the reflexive, transitive, and symmetric equivalence relation.  This can be expressed as $\sim$ and signifies that the sets are of equal size.

$$
\{g(a_i(x)) \; \forall \; i \in \Bbb N \} \sim \Bbb I \in \(0, 1\] \\
\Bbb A \sim \Bbb I
$$

Thus the outputs of the set of functions with aperiodic iterations, $\{ f_a(x) \}$ when transformed using $g(y)$ corresponds to the set of all irrational numbers $\Bbb I$.

### Rational numbers correspond to periodic functions

Conversely, take the set of all functions that map onto the interval $(0, 1]$, 

$$
\Bbb B \sim \{ h_i(x) \; \forall \; i \in \Bbb N \}
$$

Map the outputs of any element of $\Bbb B$ to a sequence of digits after $0.$ with $f(x)$ as above.  For any starting value $x_0$, the output of a single function mapped to digits, $g(h_i(x))$, is only guaranteed to yeild a periodic digit sequence for all iterations if $h_i(x)$ is periodic.  Thus 

$$
\{g(h_i(x)) \; \forall \;i \in \Bbb N \} \sim \Bbb Q \\
\Bbb B \sim \Bbb Q
$$

or in other words, the set of all periodic functions is equivalent to the set of rational numbers in the interval $(0, 1]$ by $g(h)$, and therefore also equivalent to the set of all numbers $\Bbb Q$


### Aperiodicity is equivalent to sensitivity to initial conditions

Say that points close together are stable if they stay close together in the arbitrary future, or are unstable if they diverge.  

$$
stable \Rightarrow | f(x) - f({x+\epsilon}) | \leq |f^n(x) - f^n((x+\epsilon)) \; \forall \; n \in \Bbb N|
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

### Chaotic systems are unsolveable

Let's define solvable equations to be those that are computable in finite time for all inputs.  There exists and elegent proof for the idea that nearly every decision problem (that outputs one of two options) is not computable in finite time [here](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/lecture-23-computational-complexity/).  This proof establishes that nearly all decision (output 'yes' or 'no') problems are unsolveable in finite time, and can be tweaked to establish an equivalence between uncomputable (by a finite algorithm) problems and irrational numbers as follows:

Any program to compute a decision may be represented as a binary string of finite length, which is also a representation for any natural number.  

$$ 
program \approx binary \; string \approx x \in \Bbb N
$$

Now rational numbers are just two natural numbers divided by each other so any finite program can be represented by a rational number.

$$ 
x \in \Bbb Q \;if \; x= \frac{a}{b}, \; |\; a, b \in \Bbb N \\
program \approx binary \; string \approx x \in \Bbb Q
$$

The solution to any decision problem may be denoted as an infinite string of bits, 1 for 'yes' or 0 for 'no' for any number of infinite inputs, which corresponds to any real number.  As any program to compute a decicision is finite, and as this program may be represented by a rational number then only infinite strings of bits that are also representations of rationals may be computable. All other problems are unsolvable in finite time (with a finite program), and as any member of this set of problems is not rational but in the real numbers

$$
\{unsolvable \; problems\} \sim \Bbb R - \Bbb Q \sim \Bbb I
$$

Now as we have established an approximate equivalence between irrational numbers and aperiodic (chaotic) systems above, by transitivity we can establish an approximate equivalence between the set of unsolvable decision problems and the set of all outputs of chaotic systems or 

$$ 
\Bbb A \approx \Bbb I \equivalencesim \{unsolvable \; problems\}
$$

Thus chaotic systems are unsolvable in finite time.










