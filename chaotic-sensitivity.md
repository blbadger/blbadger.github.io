### Aperiodicity implies sensitivity to initial conditions

As we have seen for the [logistic map](https://blbadger.github.io/logistic-map.html), small changes in starting values lead to large changes after many iterations.  It turns out that a fundamental feature of all chaotic systems is that their lack of periodicity implies extreme sensitivity to initial values, and this was shown by Lorenz in his pioneering work on convection.  Here follows a proof using contraposition, based on the work of Ed Lorenz.

### Theorem: Aperiodicity implies sensitivity to initial values (discrete version)

Restated, take $f$ to be a nonlinear function in finite dimensional phase space that is bounded by finite values and iterated discretely.  This entails that trajectories are unique, and that if $f$ returns to a previous point then $f$ is periodic.  Define $f$ to be aperiodic when future iterations do not revisit previous points in the space.  In symbols, an aperiodic $f$ is defined as follows:

$$
f(x) : f^n(x_0) \neq f^k(x_0) \; \forall n, k : n \neq k
$$

(An example of an $f$ satisfying these conditions is seen in the link at the top of this page, proving that this is not a vacuous statement)

And in contrast, $f$ is periodic when there is some period $f^n$ such that this future iteration is located at a current or past point:

$$
f(x) : f^n(x_0) = f^k(x_0) \; \exists n, k: n \neq k
$$

Note that these definitions restrict the following to discrete maps.  In higher dimensional continuous maps, $f(x)$ may be strictly aperiodic in that it never revisits a previous point, but this $f(x)$ may be decomposed into independant lower-dimensional trajectories that are periodic, revisiting previous points arbitrarily often.  See below for more on this subject.

**Proof:** 

There is nothing special about our initial value $x(0)$ relative to the others obtained by iterating an equation.  So any value that is iterated from $f$ can be considered a 'starting value'.  Now suppose that we make a small change to an iterated value $x_n$ to produce $x_n^*$

$$ x_n^* =  x_n + \epsilon  $$

where $\epsilon$ is an arbitrarily small finite number. Now suppose that this small change does not change future values, such that for any iteration number $i$,

$$\lvert f^i(x_n) - f^i(x_n^* ) \rvert \le \epsilon $$ 

ie $f^i(x_n)$ and $f^i(x_n^* )$ stay arbitrarily close to each other for all iterations.

As phase space trajectories are unique (meaning that any given point of the system has only one future trajectory), this system must be periodic if in is insensitive to initial values: whenever $x_{n+i}$ is within $\epsilon$ to $x_n$, the same iteration pattern obtained between these two points must repeat (recall that $f$ is defined to be bounded). The period may be very large, in that it may take many iterations of $f$ to come within $\epsilon$ of $x_n$, but if $\epsilon$ is finite then so will the period be.  

The geometric argument for this statement is as follows: if the conditions above are satisfied and we assume that future iterations $x_{n+a}$ finitely close to a current point $x_n$ travel together, there is a small but finite ball $B$ around each point on the trajectory $T$ where the radius of $B$ is $\epsilon$.  As the trajectory remains in a finite area by definition, it must revisit one of $B$ after infinite time because a finite area can be tiled by a finite number of $B$.  As $f$ is nonlinear,$B$ either grows or shrinks over time in the general case.  If $B$ is revisited, then $x_{n+a} - x_n \to 0$ as $t \to \infty$ and previous values are visited asymptotically.

Revisiting a previous value is equivalent to periodicity, and therefore insensitivity to initial values (in this case $x_n$) implies periodicity.  Taking the contrapositive of this statement, we have it that aperiodicity implies sensitivity to initial values (for discrete maps of $f$). All together, 

$$
f(x) : f^n(x(0)) \neq f^k(x(0)) \implies \\
\forall x_1, x_2 : \lvert x_1 - x_2 \rvert < \epsilon, \\
\exists n \; : \lvert f^n(x_1) - f^n(x_2) \rvert > \epsilon
$$

### Aperiodicity and sensitivity to initial values: continuous version

Lorenz defined sensitivity to initial conditions using a $\delta, \epsilon$ -style definition analagous to that for continuity.  This is especially interesting because it implies that trajectory which is sensitive to initial values is discontinuous in the output space with respect to the input. Other pages on this site show that there are too many possible aperiodic trajecotires for them to be continuous, as well as more direct proofs that aperiodicity implies discontinuity (with respect to the output space).  

### Decomposably periodic or quasiperiodic functions that are insensitive to initial values

The case for periodic versus aperiodic functions mapped continuously is mostly similar, but with a few extra considerations. The first and probably the most obvious is that with no discrete unit of time and therefore no iterations to speak of.  Instead there is a continuum, which one can think of as an infinite number of iterations between any two points in the trajectory.  Therefore rather than a finite $k$ number of iterations defining a period, there is some finite time $t$ that defines it.  Secondly, trajectories cannot cross one another's path in continuous maps, whereas they may for discrete cases where the points do not fall on top of one another.  

The theorem above extends to continuous functions in higher dimensions that do not necessarily have periodic outputs, but that may be decomposed into periodic trajectories that are independant of one another.  A simple case of this can be seen for two points rotating on circular orbit independantly of one another: if one has a rotational period of 1 and another of $\pi$, after starting in the same place on their respective circles they never again would do so because $\pi$ is irrational but 1 is rational.  Both circular orbits are periodic but the combination of the two is strictly aperiodic, but not sensitive to initial values because it can be composed of two independant periodic systems.

Note that the above argument does not apply to discrete maps.  This is because if an equation is iterated discretely, any period must have a finite number of iterations between $x_n$ occurrences.  Therefore the circle map above is only periodic if both trajectories reach the same point after a finite number of iterations.  Without loss of generality, say the first orbit has period $p$ and the second period $q$.  Then both points will be located in their initial position at iteration $k = pq$, which is finite as $p$ and $q$ are.  Therefore the map is necessarily periodic for discrete iterations.

Another example of an aperiodic map that can be decomposed into a periodic map: consider the case of a point traveling around a circle in discrete jumps, at some rational value between iterations.  This is technically an aperiodic system (as previous points are never revisited), but is clearly not sensitive to initial values because changing the starting position changes the final position by an identical amount.  This orbit is dense along the circle meaning that after an arbitrary number of iterations any point is arbitrarily close to any other.  If the coordinate system is changed such that a rational-valued rotation occurs upon each iteration, however, this system is periodic.  

This last function is often called 'quasiperiodic', meaning that while previous points are not actually revisited they are approximately revisited, to any degree of precision wished.  Lorenz defined quasiperiodicity as follows: given trajectory $P(t)$ for any $\epsilon > 0$, there exists a time $t_1 > t_0$ such that $P(t_1) - P(t_0) < \epsilon$ for all points on trajectory $P$. Clearly the point traveling along a circle in rational-size steps is quasiperiodic because given any point along the circle, one will find a future point arbitrarily nearby if one waits long enough.


For pages on this website, quasiperiodicity is included in the more general label of 'periodicity' for this reason.























