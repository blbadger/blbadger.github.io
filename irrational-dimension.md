## Irrational numbers on the real line

Two related paradoxes regarding real numbers are described and a resolution is proposed, which implies a number of interesting properties about dynamical systems.

### Lines are sequences of points, but real numbers are non-enumerable

The first paradox almost goes without saying, and is implicit in the definitions of rational and irrational numbers and lines.  A geometric line consists of an infinite number of points arranged in a sequence: $p_1$ may come after $p_2$ or before, but not both after and before.  Take a line formed from the rational numbers: all rationals may be enumerated in a list based on their 'height' (defined as the numberator plus denominator) and thus arranged along a line. This is Cantor's famous diagonal argument for the countability of rational numbers $ \Bbb Q$

$$
\Bbb Q = {0, \; 1/1, \; 1/2, \; 2/1, \; 1/3, \; 2/2 ... }
$$

The problem here is that real numbers are non-enumerable: real (specifically irrational) numbers are unable to be listed in a sequence because they are non-enumerable.  This means that any single list of real numbers will inevitably miss some because the reals are uncountably infinite.  Because of this, we cannot arrange all the real numbers on a line because doing so would imply that there is some way of arranging the real numbers in a list.  This means that if we try to construct a true real number line, we cannot include all real numbers.  As rational numbers are countably infinite but irrationals are uncountable (non-enumerable), in particular it is the set of irrational numbers that are too many to fit on any line. 

### Rationals are everywhere dense in the real line but are countable

The second paradox is closely related but is the consequence of topology and geometry rather than only enumerability.  Consider the properties of a geometric line, a one-dimensional object.  Any point on a line is bordered by at most two other points. To be precise, any point $x$ on the real number line $\Bbb R$ is arbitrarily close to at most two other points $y$ and $z$

$$
d(x, y) = 0 \\
d(y, z) = 0
$$

The real numbers $\Bbb R$ are described as existing on a line, also called the real number line $R^1$.  Now topologically the rational numbers $\Bbb Q$ are dense in the real numbers, meaning that the limit of every sequence of real numbers is a rational number.  In other words, the distance between any real number and a certain 'closest' rational number is 0.  Bearing in mind that points may be on one side or the other if arranged in a line, this means that rational numbers and irrational numbers must alternate in sequence along this line as follows:

$$
\Bbb R = {... x \in \Bbb Q, \; y \in \Bbb I, \; z \in \Bbb Q ...}
$$

because otherwise there would be nonzero distance between an abritrary rational and an irrational number in one direction.  The paradox here is that rational and irrational numbers cannot alternate because there are far more irrationals than rationals on the real number line, and alternating points necessitates equal set cardinality (even if they are both infinite).  If the points did alternate in this way, either rationals and irrationals must both be countably infinite or else both uncountably infinite. 


### Irrational numbers move around on the real line

One way to try to resolve these paradoxes could be to invoke another spatial dimension for irrational numbers, akin to the extra dimension occupied by imaginary numbers.  But this leads to a problem: rationals are no longer dense in the real numbers, because there are irrational numbers far from the rational number axis that do not contain a rational between them.

Instead, consider the idea that irrational numbers exist in a separate temporal dimension from rational numbers, meaning that irrational numbers act as though they are in motion along the real line.  This resolves the first paradox because irrationals are not able to be listed in sequence because any sequence of these numbers contains non-comparable elements, and the second paradox is addressed because a single rational number is at any one time bordered by two irrationals, but 'over time' it is bordered by an arbitrary number (an infinite number to be precise) of irrationals.

In Cantor's diagonal argument, no countable number of numbers between 0 and 1 can contain all real numbers because a number $b$ can always be generated that differs from all previously generated numbers ${a_1, a_2, a_3...}$ in at least one decimal place:

$$
a_1 = 0.a_{10}a_{11}a_{12}a_{13}a_{14}a_{15}... \\
a_2 = 0.a_{20}a_{21}a_{22}a_{23}a_{24}a_{25}... \\
a_3 = 0.a_{30}a_{31}a_{32}a_{33}a_{34}a_{35}... \\
... \\
b = 0.b_{1} b_{2}  b_{3} ... : b_1\neq a_{10}, b_2\neq a_{21}, b_3\neq a{32} ... 
$$

Now imagine we are constructing the number $b$ (which must be irrational because if it were rational, it would be in a countable set and therefore equal to one or ${a_1, a_2, a_3 ...}$) using this very method.  We start with a sequence of 0s (or any arbitrary sequence of digits) and a line representing the rational numbers $(0, 1)$, and place a pin at the origin because our number is currently:

$$
b = 0.00000...
$$

Now the first digit $b_1$ is added, which we can arbitrarily say is a 5:

$$
b = 0.50000...
$$

This is so far a rational number: one half.  But we know (see above) that the completed $b$ cannot be rational so we are not done.  We move the pin to the number $1/2$ and say that this is our starting point.  Now for a second digit $b_2$:

$$
b = 0.52000...
$$

The pin has moved to a new rational number $52/100$ but we are not done.  From here it should be clear what happens next: a third number $b_3$ is chosen

$$
b = 0.52800...
$$

and the pin moves again to a new rational $528/1000$ but we are not done

$$
b = 0.52810...
$$

And in fact we are never done, because otherwise we could convert the resultant number into a fraction.  Now at each stage the pin has moved to a new rational number but as we never finish, the pin does not cease its movement but instead hops ceaselessly from rational to rational, never settling on any of them.

### Implications for dynamical systems: aperiodicity and irrationality

Continuous functions may be defined on the rational numbers whereas discontinuous functions may only be defined on irrationals.  Consider that aperiodic maps require discontinuity (see [here](/discontinuities.md) for more), and therefore can only be defined on irrationals.  Recall that in aperiodic maps, points are everywhere unstable such that arbitrarily close initial values diverge after finite number of iterations (or after finite time if the system is continuous).  The idea that irrationals continually move provides a clear explanation as to why aperiodic maps are everywhere unstable:  a function that is defined on numbers that continually diverge will necessarily be sensitive to small changes in initial value.  

In contrast, functions that are defined on the rational numbers would not be expected to be sensitive to initial conditions, because these numbers are stationary and do not exist in this separate temporal dimension.  

### Aperiodic dynamical systems as *true* dynamical systems

Periodic maps may be described without time because they can be reduced to a periodic behavior with a remainder.  To see why this is the case, consider that time proceeds in a loop in periodic systems: if a system at $t=0$ is identical to the system at $t=20$ and $t=40$ then we may say that the system returns to its initial state every twenty seconds.  Equivalently, time proceeds in a loop that is twenty seconds long.  Any problem of determining future position is reduced to finding the remainder of the future time divided by twenty seconds.  Now say that you have observed and recorded the behavior of this periodic system for more than twenty seconds: the only task to do in order to determine the future position is to find the past position with the same remainder.  This can be done without calculating future values.

In contrast, aperiodic maps are unable to be separated from linear time in this way because they never revisit a previous state.  Now being that aperiodic (and not asymptotically periodic) maps are defined on the real numbers and not only on the rationals, we may think of the (mostly nonlinear) dynamical equations that may form aperiodic maps as 'true' dynamical equations, whereas the linear dynamical equations that result in periodic maps are simply static equations that have been re-formulated.






















