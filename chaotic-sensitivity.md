### Aperiodicity implies sensitivity to initial conditions

As we have seen for the [logistic map](https://blbadger.github.io/logistic-map.html), small changes in starting values lead to large changes after many iterations.  It turns out that a fundamental feature of all chaotic systems is that their lack of periodicity implies extreme sensitivity to initial values, and this was shown by Lorenz in his pioneering work on convection.  Why does aperiodic behavior imply sensitivity to initial values?  Here follows a proof using contraposition, based on the work of Ed Lorenz, the first person to recognize that bounded, aperiodic attractors in phase space necessarily exhibit sensitivity to initial conditions.

### Theorem: Aperiodicity (in phase space) implies sensitivity to initial values.

Proof: Take $f$ to be a function in finite dimensional phase space, iterated discretely, that is bounded by finite values.  This entails that trajectories are unique, and that if $f$ returns to a previous point then $f$ is periodic.  Assume that $f$ is aperiodic, such that future iterations do not 

In symbols, $f$ is defined as follows:

$$
f(x) : f^n(x(0)) \neq f^k(x(0)) \; \forall n, k : n \neq k
$$

(An example of an $f$ satisfying these conditions is seen in the link at the top of this page, proving that this is not a vacuous statement)

There is nothing special about our initial value $x(0)$ relative to the others obtained by iterating an equation.  So any value that is iterated from $f$ can be considered a 'starting value'.  Now suppose that we make a small change to an iterated value $x_n$ to produce $x_n^*$

$$ x_n^* =  x_n + \varepsilon  $$

where $\varepsilon$ is an arbitrarily small finite number. Now suppose that this small change does not change future values, such that for any iteration number $i$,

$$\lvert f^i(x_n) - f^i(x_n^*) \rvert \le \varepsilon $$ 

ie $f^i(x_n)$ and $f^i(x_n^*)$ stay arbitrarily close to each other for all iterations.

As phase space trajectories are unique (ie if any given point of the system has only one future trajectory), then this system must be periodic: whenever $x_{n+i}$ is within $\varepsilon$ to $x_n$, the same iteration pattern obtained between these two points must repeat (recall that $f$ is defined to be bounded). The period may be very large, in this it may take many iterations of $f$ to come within $\varepsilon$ of $x_n$, but if $\varepsilon$ is finite then so will the period be.  

The geometric argument is as follows: if the conditions above are satisfied, there is a small but finite ball $B$ around each point on the trajectory $T$ where the radius of $B$ is $\varepsilon$.  As the trajectory remains in a finite area by definition, it must revisit one of $B$ after infinite time because a finite area can be tiled by a finite number of $B$.

This means that insensitivity to initial values (in this case $x_n$) implies periodicity.  Taking the contrapositive of this statement, we have it that aperiodicity implies sensitivity to initial values (for discrete maps of $f$). All together, 

$$
f(x) : f^n(x(0)) \neq f^k(x(0)) \implies 
\forall x_1, x_2 : \lvert x_1 - x_2 \rvert < \epsilon, \; \exists n \; : \lvert f^n(x_1) - f^n(x_2) \rvert > \epsilon
$$

$\square$


