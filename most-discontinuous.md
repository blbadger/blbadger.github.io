### There are $2^{\Bbb Q} = \Bbb R$ continuous functions from $\Bbb Q \to \Bbb Q$

This theorem has been established and may be found in texts on real analysis, and one proof will suffice here.

First we define a 'unique' function to define a single trajectory in finite dimension $D$.  In other words, there is a one-to-one and onto mapping of a function to a trajectory.  The coordinates for this trajectory are defined by members of the set of rational numbers $\Bbb Q$ for any continuous function.  Now note that at any one point along a continuous trajectory, the next point is one of three options: it is slightly larger, slightly smaller, or the same. Graphically in two dimensions: 

![continuous function next value]({{https://blbadger.github.io}}misc_images/continuous_function_next.png)

and precisely, $f$ may increase by $\delta$, decrease by $\delta$, or stay the same where $delta$ is defined as 

$$ 
\lvert x_1 - x_2 \rvert < \epsilon \implies \lvert f(x_1) - f(x_2) \rvert < \delta 
$$ 

for any arbitrarily small value $\epsilon$. 

Thus for $\Bbb Q$ there are three options, meaning that the set of continuous functions is equivalent to the set of all sets of $\Bbb Q$ into $\{0, 1, 2\}$

$$
\{f\} \sim 3^{\Bbb Q} \sim \Bbb R
$$

or the set of all continuous functions defined on $\Bbb Q$ is equivalent to the set of real numbers. 

### There are $2^{\Bbb R}$ discontinuous functions from $\Bbb Q \to \Bbb Q$

Discontinuous trajectories in $Q$ may do more than increase or decrease by $\delta$ or else stay the same: the next point may be any element of $\Bbb Q$.   The size of the set of all discontinuous functions is therefore the size of the set of all subsets of continuous functions. As we have established that the set of all continuous functions from $\Bbb Q \to \Bbb Q$ is equivalent to $\Bbb R$, 

$$
(2^{\Bbb Q})^{\Bbb Q} = \Bbb R ^ {\Bbb R} = 2^{\Bbb R}
$$

### Discontinuous maps cannot be defined on the rationals

Functions are simply subsets of the Cartesian product of one set into another, meaning that a function mapping $\Bbb Q \to \Bbb Q $ is a subset of 

$$
\Bbb Q^{\Bbb Q} = 2^{\Bbb Q}
$$

Thus there can be at most $\Bbb R$ functions mapping $\Bbb Q \to \Bbb Q$, but we have already seen that the size of the set of discontinuous functions is $2^{\Bbb R}$.  This means that discontinuous functions cannot be defined on the set of rational numbers $Q$ for any finite dimension $D$.




