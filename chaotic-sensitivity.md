### Aperiodicity implies sensitivity to initial conditions

As we have seen for the [logistic map](https://blbadger.github.io/logistic-map.html), small changes in starting values lead to large changes after many iterations.  It turns out that a fundamental feature of all chaotic systems is that their lack of periodicity implies extreme sensitivity to initial values, and this was shown by Lorenz in his pioneering work on convection.  Why does aperiodic behavior imply sensitivity to initial values?  Here follows a proof using contraposition, based on the work of Ed Lorenz, the first person to recognize that bounded, aperiodic attractors in phase space necessarily exhibit sensitivity to initial conditions.

### Theorem: Aperiodicity (in phase space) implies sensitivity to initial values.

Proof: There is nothing special about our initial value relative to the others obtained by iterating an equation.  So any value that is iterated from (1) can be considered a 'starting value'.  Now suppose that we make a small change to an iterated value $x_n$ to produce $x_n^*$

$$ x_n^* =  x_n + \varepsilon  $$

where $\varepsilon$ is an arbitrarily small finite number. Now suppose that this small change does not change future values, such that for any iteration number $i$,

$$\lvert x_{n+i} - x_{n+i}^* \rvert \le \varepsilon $$ 

ie $x_{n+i}$ and $x_{n+i}^*$ stay arbitrarily close to each other for all iterations.

If the system contains unique trajectories (ie if any given point of the system has only one future trajectory), then this system must be periodic: whenever $x_{n+i}$ is within $\varepsilon$ to $x_n$, the same iteration pattern obtained between these two points must repeat. The period may be very large, in this it may take many iterations of (1) to come within $\varepsilon$ of $x_n$, but if $\varepsilon$ is finite then so will the period be.  As any ordinary differential equation contains only one independent variable (time), all trajectories are unique.  This means that insensitivity to initial values (in this case $x_n$) implies periodicity.  Taking the contrapositive of this statement, we have it that aperiodicity implies sensitivity to initial values $\square$


