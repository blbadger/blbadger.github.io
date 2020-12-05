### There are $2^{\Bbb Q} = \Bbb R$ continuous functions from $\Bbb Q \to \Bbb Q$ (1)

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

### There are $2^{\Bbb R}$ discontinuous functions from $\Bbb Q \to \Bbb Q$ (2)

Discontinuous trajectories in $Q$ may do more than increase or decrease by $\delta$ or else stay the same: the next point may be any element of $\Bbb Q$.   The size of the set of all discontinuous functions is therefore the size of the set of all subsets of continuous functions. As we have established that the set of all continuous functions from $\Bbb Q \to \Bbb Q$ is equivalent to $\Bbb R$, 

$$
(2^{\Bbb Q})^{\Bbb Q} = \Bbb R ^ {\Bbb R} = 2^{\Bbb R}
$$

### Discontinuous maps cannot be defined on the rationals (3)

Functions are simply subsets of the Cartesian product of one set into another, meaning that a function mapping $\Bbb Q \to \Bbb Q $ is a subset of 

$$
\Bbb Q^{\Bbb Q} = 2^{\Bbb Q}
$$

Thus there can be at most $\Bbb R$ functions mapping $\Bbb Q \to \Bbb Q$, but we have already seen that the size of the set of discontinuous functions is $2^{\Bbb R}$.  This means that discontinuous functions cannot be defined on the set of rational numbers $Q$ for any finite dimension $D$.

### Aperiodic maps are discontinuous (4)

Aperiodic, bounded trajectories are sensitive to inital values such that an arbitrarily small difference $\varepsilon$ bewteen $x_0$ and $x_1$ leads to a larger difference $\varepsilon^* > \varepsilon$ in $ \lvert f^n(x_0) - f^n(x_1) \rvert \; \exists n$. Sensitivity to initial values [implies](https://blbadger.github.io/aperiodic-irrationals.html) aperiodicity in bounded trajectories, as first recognized by Lorenz. Therefore aperiodic maps are necessarily sensitive to initial values.

Function $f$ is defined to be a continuous function from $\Bbb R \to \Bbb R$.  Now define a function $g$ that is equivalent to the composition of function $f \circ f \circ f ...$.  We can call $g$ the map of $f$, in that it yields the final position of an initial value $x_0$ in $\Bbb R$ after $n$ compositions. One $f$ yields a different $g$ for each composition $\circ f...$.  $g$ by definition is sensitive to initial values as above, because it reflects the final position after any number of compositions.

Corrollary: $g$ for an arbitrary aperiodic map is discontinuous

Proof: Suppose, for the sake of contradiction, that $g$ is continuous.  By definition, for all points $x_a, x_b$ and for any finite $\epsilon > 0$, there exists a $\delta > 0$ such that

$$
\lvert g(x_a) - g(x_b) \rvert < \epsilon \\
whenever \;  \lvert x_a - x_b \rvert < \delta 
$$

But if this was true of $g(x_a)$ and $g(x_b)$ for all $g$ then $\lvert g(x_a) - g(x_b) \rvert < \epsilon \; \forall \epsilon$ given a finite $\delta$ such that $\lvert x_a - x_b \rvert < \delta$.  But then if $x_a - x_b < \delta$ then $g(x_a)$ would stay arbitrarily close to $g(x_b)$ and $g(x)$ would not be sensitive to arbitrarily small changes in initial points $x_a, x_b...$.  This is a contradiction by the definition of $g$ above, and therefore $g(x)$ is discontinuous. $\square$

Finally, note that we can define the map of an aperiodic, bounded function $f$ based on $g$ for an arbitrarily large number of compositions of $f$.  Taking $g$ as our aperiodic map, the result is that aperiodic, bounded phase space maps are discontinuous.

### The set of all aperiodic maps cannot be defined on $\Bbb Q$ (5)

This results from (3) and (4).


