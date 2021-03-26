## Periodic trajectories and rational numbers

Here we establish an equivalence between the set of irrational numbers and the set of all aperiodic function trajectories (restricted to functions of discrete time intervals in finite precision in a given dimension), and conversely an equivalence between the set of rationals and periodic functions.  Then it is shown that aperiodic trajectories cannot be defined on rationals but instead require uncountably many elements, ie the set of irrational numbers.  These findings allow us form a short proof of sensitivity to initial conditions that typifies aperiodic trajectories, and also explore a connection between aperiodic trajectories and unsolveable problems. 

### The set of periodic trajectories is equivalent to the set of rational numbers

Here periodic trajectories are defined as trajectories of discrete dynamical equations in finite dimension at a given precision that eventually re-visit previous points.  Difference equations (or differential equations) that repeat previous points are periodic because there is no change in behavior over time, meaning that the trajectory from any given point at time 0 is identical to the trajectory from the same point at any other time.

As there is no universally-agreed upon notation for the set of irrational numbers, here we take the set of irrational numbers to be the real numbers that are not rational, $ \Bbb I = \Bbb R - \Bbb Q$, or equivalently

$$ \Bbb I = \{ x \; \vert \; x \in \Bbb R \land x \notin \Bbb Q\} $$

Rational numbers are expressible as fractions, whereas irrationals are not, and all real numbers are either rational or irrational.  As $\Bbb R $ is uncountably infinite but $\Bbb Q$ is countably infinite, $\Bbb I$ is uncountably infinite, meaning that the cardinality of the set of irrationals is much larger than the cardinality of rationals, or in other words nearly every possible number real number is irrational.

$$
card \; \Bbb I  >>  card \; \Bbb Q
$$

Now let's consider functions of discrete time intervals (difference equations or approximations of differential equations), the trajectories of which may be periodic or aperiodic. Aperiodicity means that values of the function $a(x)$ never are identical to values of previous iterations:

$$
a^n(x_0) \neq a^k(x_0) \; \mathbf {if} \; x, k \in \Bbb N \; \mathbf {and} \; k \neq x
$$

The set of all continuous functions with aperiodic outputs can be defined as $\Bbb A$

$$ 
A = \{a(x)\}
$$

Conversely, a periodic differential function is one which eventually does revisit a previous point

$$
p^n(x_0) = p^k(x_0) \; \mathbf {for} \; \mathbf {some} \; x, k \in \Bbb N \; \mathbf {given} \; k \neq x
$$

and the set of all continuous periodic functions can be denoted as
 
$$
P = \{ p(x) \}
$$

Note that included in this definition are eventually periodic trajectories, which in finite time become periodic even if they do not begin as such.  

Equivalence (of sets) is a property that is reflexive, symmetric and transitive just like equivalence between numbers or expressions. Equivalence can be expressed as $\sim$ and signifies that the sets are of equal size if they are finite, or that a one-to-one and onto (bijective) function can be established between the sets if they are not finite. Properties of one set may be used to inform properties of an equivalent set.

We can specify a bijective function from periodic trajectories to rational numbers as follows: for a given trajectory in finite dimensions, specify each point in coordinates $(x_1, x_2, x_3, ..., x_n)$ to an arbitrary finite precision (for example, if $x_1 = \pi$ can be specified as $x_1 = 3.14159265$).  Note that this precision is a representation, rather than a substitute, for the value of the trajectory coordinate.  Now for each time point, add the coordinates to $0.$ to yield a rational number.  For example, in two dimensions if 

$$(x_1, y_1) = (15.32, 10.15)$$

and 

$$(x_2, y_2) = (14.99, 11.1)$$

then the number yielded from these points is

$$ 0.1532101514991110 $$ 

Now being that the trajectory is periodic, the number will have digits that eventually repeat (in finite time) because future coordinates are identical to previous coordinates.  All numbers that have digits that eventually repeat (after a finite number of digits) are rational numbers, and therefore this function maps periodic trajectories of discrete time to the set of rational numbers $Q$. 

Is this function one-to-one? For trajectories of any one dimension, each individual trajectory maps to one digit sequence, and likewise each digit sequence represents only one trajectory (given our arbitrary precision) and therefore the function is one-to-one.  The function is onto as well, because any rational number can be represented with a periodic function in our dimension and precision constraints (the function is not restricted to certain digits at any time).  This means that our mapping function is bijective, and thus we can establish an equivalence between the set of all periodic trajectories of discrete time and the set of rational numbers:

$$
P \to \Bbb Q \; (bijective) \\
P \sim \Bbb Q 
$$

Now what about aperiodic functions of discrete time?  Using the same mapping function specified above, we can map these trajectories to digit sequences after $0.$ that do not repeat.  Irrational numbers are represented as digit sequences that do not repeat, so we have mapped aperiodic trajectories to irrational numbers.  By the reasoning in the last paragraph, the mapping is one-to-one and onto and therefore the set of aperiodic trajectories is equivalent to the set of irrational numbers,

$$
A \to \Bbb I \; (bijective) \\
A \sim \Bbb I
$$

Thus the set of all periodic trajectories (as defined above) is equivalent to the set of all rational numbers, whereas the set of aperiodic trajectories is equivalent to the set of irrational numbers.

### Periodic but not aperiodic trajectories may be defined on the rationals

Restated, discrete maps of continuous functions can only be defined on the rational numbers if the trajectory is periodic.  The proof that follows assumes that the function $f(x)$ is continuous, but the theorem also applies to discontinuous functions (see below).

The proof for this statement is as follows: for any aperiodic discrete trajectory of a continuous function $f(x)$, the Sharkovskii theorem states that the same $f(x)$ also contains (prime) period points of periods $1, \; 2, \; 4, \; 8, ...$.  Each point is unique for each prime period, meaning that there is at least one unique point in the domain of $x$ that satisfies each period.  We can classify each point by its period:

$$
x_0 = 1, \; x_1 = 2, \; x_2 = 4, \; x_3 = 8 ...
$$

which indexes $x_n$ over all the integers, meaning that there is a one-to-one and onto correspondance between periodic orbits and the integers.  The integers being countably infinite, there are countably infinitely many periodic orbits in the domain of $x$.  For periodic trajectories of $f(x)$, therefore, a countable set such as the rational numbers suffices.

The same argument shows that a countable number of inputs in the domain of $x$ is insufficient to yield an aperiodic trajectory, any element of this set would be mapped on a finite period.  This is a contradiction by definition, and therefore aperiodic trajectories must contain an uncountably infinite number of elements in their domain, ie the irrational numbers or reals.

This is perhaps most clearly demonstrated for the [logistic map](https://blbadger.github.io/logistic-map.html) where $r=4$: trajectories may be periodic only if they land on rational points on the line, whereas trajectories that exist on the irrationals are aperiodic.

### Sensitivity to initial values implies aperiodicity (for most trajectories)

Say that points close together are stable if they stay close together in the arbitrary future, or are unstable if they diverge.  

$$
stable \Rightarrow | f(x) - f({x+\epsilon}) | \leq |f^n(x) - f^n((x + \epsilon)) \; \forall \; n \in \Bbb N|
$$

where $\epsilon$ is an arbitrarily small number and iterations of $f(x)$ are denoted $f^n(x)$. 

Now suppose that all points are unstable, such that all points that start close together move apart after a number of iterations. This is equivalent to saying that a system is sensitive to changes in initial values.  Now being that every initial value will eventually have different outputs than the value's neighbor (for any 'neighbor size'), each initial value has a unique trajectory.  As there is an uncountably infinite number of possible starting values on the interval $(0,1]$, 

$$
| \{ x \; : \; x \in (0, 1] \} | = |\Bbb R|
$$

there is an uncountably infinite number of possible trajectories if no two trajectories are the same, because they can diverge anywhere.  

As we have above established an equivalence between the set of continuous functions with periodic trajectories and rational numbers, 

$$
\Bbb B \sim \Bbb Q
$$

and as there are countably many rational numbers, there are countably many periodic trajectories. But there are uncountably many trajectories if the system is unstable everywhere, so most cannot be periodic: there are simply too many (as there are far more objects in an uncountable set than a countable one). Thus, instability (sensitivity) at all initial points leads to (nearly all) trajectories being aperiodic.  

This too is reflected in the logistic map: values such as r=3.99 yield countably many (unstable) periodic trajectories and an uncountable number of aperiodic ones.


### The set of aperiodic trajectories is equivalent to the set of unsolvable problems

Let's define solvable equations to be those that are computable in finite time for all inputs.  There exists and elegent proof for the idea that nearly every decision problem (that outputs one of two options) is not computable in finite time [here](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/lecture-23-computational-complexity/).  This proof establishes that nearly all decision (output 'yes' or 'no') problems are unsolveable in finite time, and can be tweaked to establish an equivalence between uncomputable (by a finite algorithm) problems and irrational numbers as follows:

Any program to compute a decision may be represented as a binary string of finite length, which is also a representation for any natural number.  

$$ 
program \approx binary \; string \sim x \in \Bbb N
$$

Now rational numbers are just two natural numbers divided by each other so any finite program can be represented by a rational number.

$$ 
x \in \Bbb Q \;if \; x = \frac{a}{b}, \; |\; a, b \in \Bbb N \\
program \approx binary \; string \sim x \in \Bbb Q
$$

The solution to any decision problem may be denoted as an infinite string of bits, 1 for 'yes' or 0 for 'no' for any number of infinite inputs, which corresponds to any real number.  As any program to compute a decision is finite, and as this program may be represented by a rational number then only infinite strings of bits that are also representations of rationals may be computable. All other problems are unsolvable in finite time (with a finite program), and as any member of this set of problems is not rational but in the real numbers

$$
\{unsolvable \; problems\} \sim \Bbb R - \Bbb Q \sim \Bbb I
$$

Now as we have established an equivalence between irrational numbers and aperiodic (chaotic) systems above, by transitivity we can establish an equivalence between the set of unsolvable decision problems and the set of all outputs of chaotic systems or 

$$ 
\Bbb A \sim \Bbb I \sim \{unsolvable \; problems\}
$$

Thus there are as many aperiodic trajectories as there are unsolveable problems.










