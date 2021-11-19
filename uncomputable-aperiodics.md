## Computability and periodicity II

### Uncomputable but definable trajectories

Truly aperiodic (and bounded) trajectories are uncomputable, assuming finite memory and finite time (see [here](https://blbadger.github.io/solvable-periodicity.html) for more with this definition) which is to say that no finite computational procedure is able to accurately account for a bounded aperiodic trajectory.  All 'aperiodic' trajectories displayed in figures (as computed by programs) on this page are actually periodic, as there are a finite number of places any point may be given finite computational memory and so eventually one point must be repeated.  

### Computability and aperiodicity: a look at the logistic map for r=4

Recalling the logistic map, explored [here](https://blbadger.github.io/logistic-map.html)

$$
x_{n+1} = rx_n(1-x_n) \tag{1}
$$

Because of this, one would never know that the period 3, 5, 6, etc. windows in (1) only exist for an infinitely small fraction of starting points for the arbitrarily precise 'true' logistic map, or in other words that this true logistic map is quite different than the approximation presented here.

Given any binary number $\theta$, say $0.1101001$, the number's binary shift map is as follows:

$$
\theta_{n+1} = 2\theta_n \bmod 1
$$

The first few iterations of this map on $1.1101001$ are

$$
\theta_0 = 0.1101001 \\
\theta_1 = 0.1010010 \\
\theta_2 = 0.0100100 \\
\theta_3 = 0.1001000 \\
$$

Now for any rational starting number $\theta_0 \in \Bbb Q$, the bit shift map is periodic or eventually periodic because after a finite number of iterations, the remaining digits are composed of repeating sequences.  On the other hand, the bit shift map is aperiodic for $\theta_0 \in \Bbb R - \Bbb Q$.  This map also exhibits sensitivity to initial conditions because a small change in $\theta_0$ becomes exponentially larger over time, specifically $2^n$ as large after $n$ iterations.

It is surprising that we can find a solution to the logistic map for the special case where r=4 using the following mapping

$$
x_n = \sin^2 (\pi \theta_n)
$$

If $x_{n+1} = 4x_n(1-x_n)$ is implied by $x_n = \sin^2(\pi\theta_n)$ given $\theta_{n+1} = 2 \theta \bmod 1$, the latter is a solution to the logistic map.

because this map is identical to $x_{n+1}$ in the logistic map for $\theta_{n+1}$

$$
x_{n+1} = 4 \sin^2(\pi \theta_n) \left(1-\sin^2(\pi \theta_n) \right), \\
x_{n+1} = \sin^2(\pi2\theta_n \bmod 1 ) \implies \\
4\sin^2(\pi\theta_n)\cos^2(\pi\theta_n) = \sin^2(\pi2\theta_n \bmod 1 ) \\
2 \sin(\pi\theta_n)cos(\pi\theta_n) = \sin(\pi2\theta_n \bmod 1) \\
$$

and as $\sin(2\theta) = 2\sin(\theta)\cos(\theta)$, 

$$
2 \sin(\pi\theta_n)\cos(\pi\theta_n)  = 2 \sin(\pi\theta_n)\cos(\pi\theta_n) 
$$

and therefore these expressions are equivalent regardless of a choice of $\theta_0$.

Expressed another way, the solution to the logistic map with r=4 is 

$$
x_n = \sin^2(\pi 2^n \theta) 
$$

### Other solutions to aperiodic systems

The logistic map is not the only place where one can find the seemingly nonsensical conjunction of an aperiodic system being solved in a cosed-form expression.  Perhaps the two most famous of all irrational numbers, whose digits form aperiodic sequences, may be expressed using periodic procedures as follows:

$$
e = \sum_{n = 0}^\infty \frac{1}{n!} \\
\; \\
\frac{\pi^2}{6} = \sum_{n = 1}^\infty \frac{1}{n^2}
$$

The computational procedure in either case may be expressed as a simple for loop that adds together the resulting number upon substituting the loop number for $n$. But this seems somewhat counterintuitive: aperiodic sequences are inherently unpredictable, so how can they be described with a short computational procedure?  An oblique restatement is as follows: aperiodic systems are for most purposes indistinguishable from 'random' ones.  It is known that no truly random output can be made from any classical computational procedure, so how then could a (very small) classical computational procedure yield a random-like output?

If one is not convinced by the argument that aperiodic outputs are not distinguishable from random ones, consider the memory requirements of storing sequences of integers.  It is considered to be accepted that no irrational number $x \in \Bbb R - \Bbb Q$ can be stored in finite memory, and therefore no program is capable of converting an irrational number into a finite sequence of integers. Because all programs themselves may be mapped to a finite sequence of digits in a bijective manner, no aperiodic sequence can be represented by a finite program and there is no way to represent an aperiodic sequence with a periodic one.  But then how can we express irrationals like $e$ or $\pi$ or even aperiodic sequences like the logistic map for r=4 with such small periodic procedures?

The answer is that these procedures are not finite: to calculate the exact value of $e$, for example, one needs to add an infinite number of $\frac{1}{n!}$ terms together.  A similar statement is true for $\frac{\pi^2}{6}$, and upon close inspection this is also the case for the logistic map where $r=4$.  To see why this is the case, first recall that only irrational values of $\theta$ are aperiodic, and that the logistic map for $r=4$ is sensitive to initial conditions because $\theta$ is transformed by $2^n$, leading to any small change growing exponentially before the value is folded into $(0, 1)$ by $sin^2(x)$.  This means that as $n$ increases in size, the number of digits $\theta$ is known to must also increase to yield a reasonably accurate value of $x_n$.  An arbitrarily large value of $n$ requires an arbitrarily good approximation of $\theta$.

These considerations suggest that it is helpful to define three basic types of computations:

1. Numbers (inputs) grow but the procedure is of fixed size.
2. Numbers (inputs) and procedure size both grow, but the procedure itself is periodic.
3. Numbers and procedure size grow, and the procedure cannot be represented as a periodic sequence.

These classifications correspond to the notions of

1. Rational numbers: $1/2$ etc.
2. Classically computable but irrational numbers: $e,\; \pi,\; \sqrt2$ and so on.
3. Classically uncomputable (irrational) numbers: Chaitlin's constant $\Omega_f$ and others

With this classification, any aperiodic sequence falls in the second or third group. Of note, this group includes the sequence of [prime gaps](https://blbadger.github.io/unpredictable-primes.html), which yields the conclusion that that sequence, and therefore the number of integers one has to pass over to arrive at the subsequent prime number, can never be represented by a finite periodic computational procedure.  In computer science parlance, this means that finding a prime number will never be brought down to $O(1)$, but instead the procedure itself scales with input size.  It is clear that this problem is an example of type 2 computability, because the procedure for finding primes is itself periodic (simply iterate over all numbers less than or equal to $\sqrt n$ and see if any divides $n$).

With this classification, consider the logistic map where $r=4$: the case where an arbitrary aperiodic trajectory of the logistic map for $r=4$.  It is interesting to note that the closed solution presented above is only helpful for a small subset of possible starting points $x_0 \in (0, 1)$.  This is because most real numbers are classically uncomputable (a result of the diagonal argument for procedures) and therefore approximating $\theta$ to an arbitrary degree of accuracy by a (finite) computational procedure is impossible.  

Motivated by these observations, it may be best to consider a more strict definition of computability than is normally used.  For the logistic map where $r=4$, as $n \to \infty$ for an accurate computation the precision of theta must be known to an infinite number of decimal places, which is clearly not possible.  Likewise, the exact values of $e$ and $\pi$ cannot be determined without an infinite number of additions performed, also not possible.






























