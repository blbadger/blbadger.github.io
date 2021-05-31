### Uncomputable but definable trajectories

Truly aperiodic (and bounded) trajectories are uncomputable, assuming finite memory and finite time (see [here](https://blbadger.github.io/solvable-periodicity.html) for more with this definition) which is to say that no finite computational procedure is able to accurately account for a bounded aperiodic trajectory.  All 'aperiodic' trajectories displayed in figures (as computed by programs) on this page are actually periodic, as there are a finite number of places any point may be given finite computational memory and so eventually one point must be repeated.  Because of this, one would never know that the period 3, 5, 6, etc. windows in the logistic map only exist for an infinitely small fraction of starting points for the arbitrarily precise 'true' logistic map, or in other words that this true logistic map is quite different than the approximation presented here.

### Computability and aperiodicity: a look at r=4

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

and as $\sin(2\theta)) = 2\sin(\theta)\cos(\theta)$, 

$$
2 \sin(\pi\theta_n)cos(\pi\theta_n)  = 2 \sin(\pi\theta_n)cos(\pi\theta_n) 
$$

and therefore these expressions are equivalent regardless of a choice of $\theta_0$.

Expressed another way, the solution to the logistic map with r=4 is 

$$
x_n = \sin^2(\pi 2^n \theta) 
$$
