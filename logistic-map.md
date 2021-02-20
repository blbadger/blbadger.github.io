## Logistic Map

### Introduction: periodic trajectories in the logistic map

The logistic map was derived from a differential equation describing population growth, studied by Robert May. The dynamical equation is as follows:

$$x_{n+1} = rx_n (1 - x_n) \tag{1}$$

where r can be considered akin to a growth rate, $x_{n+1}$ is the population next year, and $x_n$ is the current population.  Population ranges between 0 and 1, and signifies the proportion of the maximum population.

Let's see what happens to population over time at a fixed r value.  To model (1), we will employ numpy and matplotlib, two indispensable python libraries.

```python
#! python3
# import third-party libraries
from matplotlib import pyplot as plt 
import numpy as np 

# initialize an array of 0s and specify starting values and r constant
steps = 50
x = np.zeros(steps + 1)
y = np.zeros(steps + 1)
x[0], y[0] = 0, 0.4

r = 0.5

# loop over the steps and replace array values with calculations
for i in range(steps):
	y[i+1] = r * y[i] * (1 - y[i])
	x[i+1] = x[i] + 1

# plot the figure!
fig, ax = plt.subplots()
ax.plot(x, y, alpha=0.5)
ax.set(xlabel='Time (years)', ylabel='Population (fraction of max)')
plt.show()
```

When $r$ is small (0.5), the population heads towards 0:

![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/logistic_time_r0.8.png)


As $r$ is increased to 2.5, a stable population is reached:

![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/logistic_time_r2.5.png)

This is called a period one trajectory, and occurs when $x_{n+1} = x_n$, so at $r=2.5$ we have

$$
x = 2.5x(1-x) \\
0 = -2.5x^2 + 1.5x = x(-\frac{5}{2}x+\frac{3}{2}) \\
x_n = 0, \; x = 3/5 \\
$$

where the value of $x_{n+1} = 3/5$ is an attractor for all initial values in $(0, 1)$ and $x_{n+1} = 0$ only occurs for $x_0 = 0$. Stability for one dimensional systems may be determined using linearization, and for iterated systems like the logistic map this involves determining whether or not $\lvert f'(x) \rvert < 1$.  If so, the point is stable but if $\lvert f'(x) \rvert > 1$, it is unstable (and if it equals one, we cannot say).  For the logistic equation, $f'(x) = r-2rx$ and so at $r=2.5$,

$$
f'(x) = r - 2rx \\
f'(x) = 3.5-7x \\
f'(0) = 3.5-0 > 1
$$

and therefore $x=0$ is unstable.  On the other hand, $f'(3/5) = 3.5-4.2 = -0.7$ which means that $x=3/5$ is a stable point of period 1.  As we shall see below, stable points of finite period act as attractors, such that points in $(0, 1)$ eventually end up on the point $x=3/5$ given sufficient iterations.

If $r = 3.1$, the population fluctuates, returning to the starting point every other year.  This is called 'period 2':

![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/logistic_time_r3.1.png)

We can find points where $x_{n+1} = x_n$, 

$$
x = rx_n(1-x)  \\
0 = x(r-1 - rx) \\
0 = x(2.1-3.1x) \\
x = 0, x \approx 0.6774...
$$

but some quick numerical experimentation shows that these points are unstable: any slight deviation from the values (and indeed any finite approximation of 0.6774...) will not result in $x_{n+1} = x_n$.  

In contrast, the points of period 2 may be found as follows:

$$
f(f(x)) = f^2(x) = x \\
x = r^2x(1-x)(1-rx(1-x)) \\
0 = x(-r^3x^3 + 2r^3x^2 - (r^3+r^2)x + (r^2-1))
$$

which when $r=3.1$ has a root at $x=0$.  The other roots may be found using the rather complicated cubic equation, and are $x\approx 0.76457, x \approx 0.55801, x \approx 0.67742$.  Note that the two unstable period-1 points are included (see below), but that there are two new points.  To see if they are stable,  we can check if $\lvert (f^2)' \rvert < 1$, and indeed as $(f^2)'(x) = -4r^3x^3 + 6r^3x^2 - 2\left(r^3+r^2\right)x + r^2$, and substituting both $x=0.55801$ and $x=0.76457$ yields a number around $\lvert 0.5899.. \rvert < 1$, meaning that both points are stable.  Therefore both are attractors, as seen in the previous numerical map.  On the other hand, $(f^2)'(0.67742) = 1.21$ and $(f^2)'(0) = 9.61$, neither of which are between positive and negative one and thus both points are unstable.  The point $x\approx 0.67742$ would be stable if the period had not increased: this is a period-doubling bifurcation. 

Note that 'period' on this page signifies what is elsewhere sometimes referred to as 'prime period', or minimal period.  Take a fixed point, for example at $r=2.5$ this was found to be located $x=3/5$.  This has period 1 because $x_{n+1} = x_n$, but it can also be thought to have period 2 because $x_{n+2} = x_n$ and period 3, and any other finite period because $x_{n+k} = x_n \forall k$.  This is why we found period-1 points when trying to compute period-2 values above.

But there is a clear difference between this sort of behavior and that where $x_{n+2} = x_n$ but $x_{n+1} \neq x_n$, where the next iteration does not equal the current but two iterations in the future does.  The last sentence is true for the logistic map where $r=3.1$, and we can call this 'prime period 2' to avoid ambiguity.  But for this and other pages on this site, 'prime' is avoided as any specific value referred to by 'period' is taken to mean 'prime period'.  

at $r = 3.5$, the trajectory is period 4, as it takes 4 iterations for the population to return to its original position:

![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/logistic_time_r3.5.png)

and at $r=3.55$, the population trajectory is period 8:

![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/logistic_time_r3.55.png)

### Aperiodic trajectories in the logistic map

For $r=3.7$, the (prime) period is longer than the iterations plotted and is actually infinite, and therefore the system is called aperiodic.  To restate, this means that previous values of the logistic equation are never revisited for an aperiodic $r$ value.  The ensuing plot has points that look random but are deterministic.  The formation of aperiodic behavior from a deterministic system is termed mathematical chaos.

![map]({{https://blbadger.github.io}}/logistic_map/logistic_time_r3.7.png)

Are there any non-prime periodic points?  Looking for points of period 1, we find two:

$$
x_n = rx_n(1-x_n) \implies 0 = x_n(r-1-rx_n) \\
0 = x_n(2.7-3.7x_n) \\
x_n = 0, \; x_n = 27/37
$$

Both are unstable, as $f'(0) = 3.7 - 2*3.7*0 > 1$ and $f'(27/37) = 3.7 - 54/37 > 1$.  For points of period 2, 

$$
0 = x_n(-r^3x_n^3 + 2r^3x_n^2 - (r^3+r^2)x + (r^2-1)) \\
x_n = 0, \; x_n = 27/37 \; x_n \approx 0.88, \; x_n \approx 0.39
$$

the fixed points are found again, and in addition two more points are found.  All are unstable (which can be checked by observing that all points fail the test of $\lvert (f^2)'(x_n) \rvert < 1$).  What makes the $r=3.7$ value special is that for any given integer $k$, one can find periodic points with period $k$.  But as $k+1$ also exhibits periodic points, the logistic map with $r=3.7$ only has a prime periodic point at $x_n = \infty$, which is to say that the map has no finite (prime) periodic points and therefore aperiodic.

Aperiodicity leads to unpredictable behavior.  Ranging $r=3.5 \to r=4$, small changes in $r$ lead to little change to iterations of (1) with $x_0 = 0.3$ if the trajectory is periodic.  But when aperiodic, small changes in $r$ lead to large changes in the population trajectory.

![map]({{https://blbadger.github.io}}/logistic_map/logistic_pop.gif)

As demonstrated by Lorenz in his [pioneering work on flow](https://journals.ametsoc.org/doi/abs/10.1175/1520-0469(1963)020%3C0130:dnf%3E2.0.CO;2), nonlinear dissipative systems capable of aperiodic behavior are extremely sensitive to initial conditions such that long-range behavior is impossible to predict.    

Observe what happens when the starting population proportion is shifted by a factor of one ten-millionth with $\Delta r=3.7$:

![map]({{https://blbadger.github.io}}/logistic_map/logistic_time_r3.7_comp.png)


The behavior is similar to the unshifted population for a while, until it changes and becomes wildly different.  This sensitivity to initial conditions, and has been shown by Lorenz to be implied by and to imply aperiodicity (more on this below).

In contrast, a relatively large shift of a factor of one hundreth (3 to 3.03) in initial population leads to no change to periodicity or exact values at $r=2.5$ (period 1):

![map]({{https://blbadger.github.io}}/logistic_map/logistic_time_2.5_hundreth.png)


or at $r=3.5$, period 4, the same change does not alter the pattern produced:

![map]({{https://blbadger.github.io}}/logistic_map/logistic_time_3.5_hundreth.png)


even a large change in starting value at $r=3.55$ (period 8), from population $p=0.3$ to $p=0.5$ merely shifts the pattern produced over by two iterations but does not change the points obtained or the order in which they cycle:

![map]({{https://blbadger.github.io}}/logistic_map/logistic_large.png)


### A closer look at aperiodicity with an orbit map

Information from iterating (1) at different values of $r$ may be compiled in what is called an orbit map, which displays the stable points at each value of $r$.  These may also be though of as the roots of the equation with specific $r$ values. 

To do this, let's have the output of the logistic equation on the y-axis and the possible values of $r$ on the x-axis, incremented in small units.  
```python
#Import third-party libraries
import numpy as np 
import matplotlib.pyplot as plt 

def logistic_map(x, y):
	'''a function to calculate the next step of the discrete map.  Inputs
	x and y are transformed to x_next, y_next respectively'''
	y_next = y * x * (1 - y)
	x_next = x + 0.0000001
	yield x_next, y_next
```
For high resolution, let's use millions of iterations.  Note that the logistic equation explodes to infinity for $r > 4$, so starting at $r = 1$ let's use 3 million iterations with step sizes of 1/100,000 (as above)
```python
steps = 3000000

Y = np.zeros(steps + 1)
X = np.zeros(steps + 1)

X[0], Y[0] = 1, 0.5

# map the equation to array step by step using the logistic_map function above
for i in range(steps):
	x_next, y_next = next(logistic_map(X[i], Y[i])) # calls the logistic_map function on X[i] as x and Y[i] as y
	X[i+1] = x_next
	Y[i+1] = y_next
```

Let's plot the result! A dark background is used for clarity.
```python
lt.style.use('dark_background')
plt.figure(figsize=(10, 10))
plt.plot(X, Y, '^', color='white', alpha=0.4, markersize = 0.013)
plt.axis('on')
plt.show()
```

![map]({{https://blbadger.github.io}}/logistic_map/logistic_period.png)


By looking at how many points there are at a given $r$ value, the same patter of period doubling may be observed. The phenomenon that periodic nonlinear systems become aperiodic via period doubling at specific ratios was found by Feigenbaum to be a [near-universal feature](https://www.ioc.ee/~dima/mittelindyn/paper4.pdf) of the transition from periodicity to chaos.

Let's take a closer look at the fuzzy region of the right. This corresponds to the values of $r$ which are mostly aperiodic, but with windows of periodicity.  There are all kinds of interesting shapes visible even in the aperiodic (fuzzy) sections, highlighting a key difference between mathematical chaos and the usual English usage (OED: a state of complete confusion and lack of order).  

![map]({{https://blbadger.github.io}}/logistic_map/logistic_period_zoom2.png)

What do these shapes mean? It is worth remembering what this orbit diagram represents: a collection of single iterations of (1) with very slightly different $r$ values, the previous iteration population size being the input for the current iteration. This is why the chaotic regions appear to be filled with static: points that are the result of one iteration of the logistic equation are plotted, but the next point is mostly unpredictable and thus may land anywhwere within a given region.  The shapes, ie regions of higher point density, are values that are more common to iterations of changing $r$ values.

Are these same values more common if $r$ is fixed and hundreds of iterations are performed at various starting population size values? Let's take $ r \approx 3.68$, where the orbit diagram exhibits higher point density at population size $p \approx 0.74$.  If we count the number of iterations near each population value using R as follows,

```R
# data taken from 900 iterations of the logistic equation starting at 3, 6, and 9 (+ 0.0000000000000001) at r=4
library(ggplot2)
data1 = read.csv('~/Desktop/logistic_0.3_4.csv', header=T)
a <- ggplot(data1, aes(value))

a + geom_dotplot('binwidth' = 0.01, col='blue', fill='red') +
    xlim(0, 1) +
    theme_bw(base_size = 14) 
```

we find that there are indeed more iterations that exist near $0.74$. Here the x-axis denotes the population size, and the y-axis denotes the number of iterations:

![map]({{https://blbadger.github.io}}/logistic_map/logistic_probs_3.68.png)


This also holds for $r = 4$: at this value, the orbit diagram suggests that there is more point density at population size $p=1$ and $p=0$ than anywhere else.  Is this the case while holding $r$ constant and iterating many times over different starting population values? It is indeed! 

![map]({{https://blbadger.github.io}}/logistic_map/logistic_probs_4.png)

Why are certain points more common than others, given that these systems are inherently unpredictable?  Iterating (1) at $r=3.68$ as shown above provides an explanation: the population only slowly changes if it reaches $p \approx 0.74$ such that many consecutive years (iterations) contain similar population values.

![map]({{https://blbadger.github.io}}/logistic_map/logistic_time_3.68.png)

The idea that the orbit map reflects the behavior of iterations of (1) at constant $r$ values implies another difference between mathematical chaos and true disorder. Consider two points, $r = 3.6$ and $r = 4$, and observe the points plotted at both values on the orbit map:

![map]({{https://blbadger.github.io}}/logistic_map/logistic_period_zoom2.png)

Both values of $r$ lead to aperiodic behavior (for almost all starting values if $r=4$), but there is a notable difference in the potential range of population sizes reached at each value: it appears that at $r=3.6$, population values are restricted to two regions (around 0.3 to 0.6 and around 0.8 to 0.9), whereas at $r=4$ the population values span the entire interval $(0, 1]$.  

Also in contrast to complete disorder, short-range prediction is possible with chaotic systems even if long-range prediction is impossible (see above).  Does relatively restricted aperiodicity (as seen for $r=3.6$) lead to an extension in prediction range?  Let's compare iterations of two starting values at a factor of a ten-thousanth apart (0.3 and 0.30003) to find out:

$r=3.6$
![map]({{https://blbadger.github.io}}/logistic_map/logistic_time_3.6_small.png)

$r=4$
![map]({{https://blbadger.github.io}}/logistic_map/logistic_time_4_small.png)

Observe that the divergence in values occurs later for $r=3.6$ than for $r=4$, implying that longer-range prediction is possible here.  Iterations of (1) at both values of $r$ are chaotic, but they are not equally unpredictable.

To illustrate this more clearly, here is a plot of the first iteration of divergence (100 iterations mazimum) of (1) at varying $r$ values, with $p_{01} = 3, p_{02} = 3.0003$.
To do this, the first step is to initialize an array to record $r$ values from 3.5 to 4, taking 500 steps.
```python
from matplotlib import pyplot as plt 
import numpy as np 

steps = 500
y = np.zeros(steps + 1)
r = np.zeros(steps + 1)
r[0] = 3.5
y[0] = 100
```
THe next step is to count the number of iterations it takes to diverge (with a maximum interation number of 100 in this case).  Arbitrarily setting divergence to be a difference in value greater than 0.15, 
```python
for i in range(500):
	X1 = 0.3
	X2 = 0.3 + 0.3**(t/30)

	for j in range(100):
		X1 = r[i] * X1 * (1-X1)
		X2 = r[i] * X2 * (1-X2)

		if np.abs(X1 - X2) > 0.15:
			break

	y[i+1] = j + 1
	r[i+1] = r[i] + 0.001
```
and now plot the results,
```python
fig, ax = plt.subplots()
ax.plot(r, y)
ax.set(xlabel='r value', ylabel='First iteration of divergence')
plt.show()
plt.close()
```

which yields

![map]({{https://blbadger.github.io}}/logistic_map/logistic_divergence_3.0003.png)

Prediction ability (in length until divergence) tends to decrease with increasing $r$, but the exact relationship is unpredictable: some small increases in $r$ lead to increased prediction ability.

Increasing the accuracy of the initial measurement would be expected to increase prediction ability for all values of $r$ for (1).  Is this the case? Let's go from $\Delta x_0 = 1 \to \Delta x_0 \approx 3.5 \times 10^{-11}$.  

![map]({{https://blbadger.github.io}}/logistic_map/logistic_divergence.gif)

Prediction power does increase with better initial measurements, but not always: the benefit is unpredictable.  Notice that for certain values of $r$ the number of iterations until divergence actually increases with a decrease in $\Delta x_0$: this means that paradoxically increased accuracy can lead to decreased prediction accuracy!  

In the real world, no measurement is perfect but is merely an estimation: the gravitational constant is not 9.8 meters per second squared but simply close to this value.  Necessarily imperfect measurements mean that not only would one have to take infinitely long to predict something at arbitrarily (infinitely) far in the future, but beyond a certain point the predictions will be inaccurate.  

This was first shown in Lorenz's [pioneering work](https://journals.ametsoc.org/view/journals/atsc/20/2/1520-0469_1963_020_0130_dnf_2_0_co_2.xml?tab_body=pdf) mentioned above, in which a deterministic model of air convection was observed to exhibit unpredictable behavior.  A simplified proof for the idea that aperiodicity implies sensitivity to initial values, based on Lorenz's work, is found [here](https://blbadger.github.io/chaotic-sensitivity.html). 

Let's call locations where close-together points eventually diverge in time unstable points.  Chaotic systems are unstable everywhere, meaning that any trajectory initially close to $x_n$ will in time diverge as $n \to \infty$.  

But such systems are not necessarily equally unstable everywhere, and the iterations at $r=3.68$ in the last section provide a graphical example of a certain value ($x_n \approx 0.74$) than others.  This illustrates a feature of chaos that differs from its English usage: mathematical chaos is not completely disordered. A more descriptive word might be 'mysterious' because these systems are unpredictable, even if they are partially ordered or are bounded by spectacular patterns, as seen in the following section.

### The relationship between aperiodic systems and fractals

One of the most striking features of this map is that it is a self-similar fractal.  This means that smaller parts resemble the whole object.

![map]({{https://blbadger.github.io}}/logistic_map/logistic_period_zoom2.png)

Observe what happens when we zoom in on the upper left section: A smaller copy of the original image is found.

![map]({{https://blbadger.github.io}}/logistic_map/logistic_period_zoom3.png)

If we take this image and again zoom in on the upper left hand corner, again we see the original!

![map]({{https://blbadger.github.io}}/logistic_map/logistic_period_zoom4.png)

Zooming in by a factor of $2^{15}$ on the point $(3.56995, 0.8925)$, we have

![map]({{https://blbadger.github.io}}/logistic_map/logistic_zoom2.gif)

An infinite number of smaller images of the original are found with increasing scale.  Far from being unusual, the formation of fractals from nonlinear systems is the norm provided that they are dissipative and do not explode towards infinity nor exhibit a point or line attractor.  

### Patterned entrances and exits from chaos

As $r$ increases, the periodicity increases until it becomes infinite, and infinite periodicity is equivalent to aperiodicity.  This occurs via period doubling: as can be seen clearly from the logistic map, one period splits into two, which split into four, which split into eight etc.  This period doubling occurs with less and less increase in $r$ such that an infinite period is reached within a finite increase in $r$, and at this point the map is aperiodic. This occurs whenever there is a transition from periodicity to aperiodicity.  

Looking closely at the orbit map, it seems that there are regions where an aperiodic trajectory transitions into a periodic one, for example at around $x = 3.83$ where there are three values where that $f(x)$ is located.  Is this really a transition away from aperiodicity?

Surprisingly, no..


There is also a pattern here: the convergence of 'favored' values tangentially leads to a transition from chaotic, aperiodic iteration to periodic. Note that the following is only conjectural.

Recall that areas with higher point density correspond to population values that appear more often over many iterations.  With increases in $r$, these populations that are more probable, the 'favored' populations, change.  There is always more than one favored population size, and occasionally with increases in $r$ two different favored sizes can converge to become one.  If the difference between two favored sizes goes to 0 at a decreasing rate, increasing $r$ leads to periodicity from aperiodicity. 

This can be clearly seen with a look at the map: if dense areas in chaotic regions approach but do not cross, there is a region of periodicity immediately following.  If these lines approach and cross (if they approach at a rate that does not decrease to 0), then aperiodicity remains for subsequent, slightly larger values of $r$. 

![map]({{https://blbadger.github.io}}/logistic_map/logistic_zoom_4.png)

In other words, let's call the set favored values, $\mathbf A$, obtained for any value of $r$ while iterating (1) to be

$$\mathbf A = \{x_1, x_2, ..., x_i\}$$ 

where $i$ is the index of the favored value. 

For any pair of elements $(x_i, x_j) \in \mathbf A$, if 

$x_i - x_j \to 0$ and $\frac{d}{dr} (x_i - x_j) \to 0$ as $r$ increases, 

then successively larger values of $r$ move (1) from aperiodic to periodic iterations.  These periodic iterations are found at population sizes equal to the value of $x_i$ as $x_i - x_j \to 0$.  It also appears that if there is (tangential) intersection between any element $x_i \in \mathbf A$ and and either the maximum or minimum possible value of (1) at any given $r$, then further increases of $r$ will lead to periodicity.  


