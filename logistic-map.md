## Logistic Map

The logistic equation was derived from a differential equation describing population growth, studied by Robert May. The equation is as follows:

$$x_{n+1} = rx_n (1 - x_n) \tag{1}$$

where r can be considered akin to a growth rate, $x_{n+1}$ is the population next year, and $x_n$ is the current population.  Population ranges between 0 and 1, and signifies the proportion of the maximum population.

Let's see what happens to population over time at a fixed r value.  To model (1), we will employ numpy and matplotlib, two indispensable python libraries.

```python
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


If $r = 3.1$, the population fluctuates, returning to the starting point every other year.  This is called 'period 2':

![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/logistic_time_r3.1.png)


at $r = 3.5$, the population is period 4, as it takes 4 iterations for the population to return to its original position:

![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/logistic_time_r3.5.png)


and at $r=3.55$, the population is period 8:

![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/logistic_time_r3.55.png)


and at $r=3.7$, the period is longer than the iterations plotted (actually it is infinite).  The ensuing plot has points that look random but are deterministic.  The formation of aperiodic behavior from a deterministic system is called mathematical chaos.

![map]({{https://blbadger.github.io}}/logistic_map/logistic_time_r3.7.png)


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


### A closer look at chaos with an orbit map

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


### Aperiodicity implies sensitivity to initial conditions

A fundamental feature of chaotic systems is that their lack of periodicity implies extreme sensitivity to initial values, and this was shown by Lorenz in his pioneering work on convection.  Why does aperiodic behavior imply sensitivity to initial values?  

There is nothing special about our initial value relative to the others obtained by iterating an equation.  So any value that is iterated from (1) can be considered a 'starting value'.  Now suppose that we make a small change to an iterated value $x_n$ to produce $x_n^*$

$$ x_n^* =  x_n + \varepsilon  $$

where $\varepsilon$ is an arbitrarily small finite number. Now suppose that this small change does not change future values, such that for any iteration number $i$,

$$\lvert x_{n+i} - x_{n+i}^* \rvert \le \varepsilon $$ 

ie $x_{n+i}$ and $x_{n+i}^*$ stay arbitrarily close to each other for all iterations.

If the system contains unique trajectories (ie if any given point of the system has only one future trajectory), then this system must be periodic: whenever $x_{n+i}$ is within $\varepsilon$ to $x_n$, the same iteration pattern obtained between these two points must repeat. The period may be very large, in this it may take many iterations of (1) to come within $\varepsilon$ of $x_n$, but if $\varepsilon$ is finite then so will the period be.  As any ordinary differential equation contains only one independent variable (time), all trajectories are unique.  This means that insensitivity to initial values (in this case $x_n$) implies periodicity.  Taking the contrapositive of this statement, we have it that aperiodicity implies sensitivity to initial values $\square$

Let's call locations where close-together points eventually diverge in time.  Chaotic systems are unstable everywhere, meaning that any trajectory initially close to $x_n$ will in time diverge as $n \to \infty$.  

But such systems are not necessarily equally unstable everywhere, and the iterations at $r=3.68$ in the last section provide a graphical example of a certain value ($x_n \approx 0.74$) than others.  This illustrates a second feature of chaos that differs from its English usage: mathematical chaos is not completely unordered. A more descriptive word might be 'mysterious' because these systems are unsolveable and unpredictable, even if they are partially ordered or are bounded by spectacular patterns, as seen in the next section.


### The relationship between chaotic systems and fractals

One of the most striking features of this map is that it is a self-similar fractal.  This means that smaller parts resemble the whole object.

![map]({{https://blbadger.github.io}}/logistic_map/logistic_period_zoom2.png)

Observe what happens when we zoom in on the upper left section: A smaller copy of the original image is found.

![map]({{https://blbadger.github.io}}/logistic_map/logistic_period_zoom3.png)

If we take this image and again zoom in on the upper left hand corner, again we see the original!

![map]({{https://blbadger.github.io}}/logistic_map/logistic_period_zoom4.png)

An infinite number of smaller images of the original are found with increasing scale.  Far from being unusual, the formation of fractals from nonlinear systems is the norm provided that they are dissipative and do not explode towards infinity nor exhibit a point or line attractor.  

### Patterned entrances and exits from chaos

As $r$ increases, the periodicity increases until it becomes infinite, and infinite periodicity is equivalent to aperiodicity.  This occurs via period doubling: as can be seen clearly from the logistic map, one period splits into two, which split into four, which split into eight etc.  This period doubling occurs with less and less increase in $r$ such that an infinite period is reached within a finite increase in $r$, and at this point the map is aperiodic. This occurs whenever there is a transition from periodicity to aperiodicity, which can be most clearly seen if we focus on a smaller region of chaotic behavior:

![map]({{https://blbadger.github.io}}/logistic_map/logistic_zoom3.png)


What about the transition from aperiodicity back to periodicity? There is also a pattern here: the convergence of 'favored' values tangentially leads to a transition from chaotic, aperiodic iteration to periodic. 

Recall that areas with higher point density correspond to population values that appear more often over many iterations.  With increases in $r$, these populations that are more probable, the 'favored' populations, change.  There is always more than one favored population size, and occasionally with increases in $r$ two different favored sizes can converge to become one.  If the difference between two favored sizes goes to 0 at a decreasing rate, increasing $r$ leads to periodicity from aperiodicity. 

This can be clearly seen with a look at the map: if lines of higher density in chaotic regions approach but do not cross, there is a region of periodicity immediately following.  If these lines approach and cross (if they approach at a rate that does not decrease to 0), then aperiodicity remains for subsequent, slightly larger values of $r$. 

![map]({{https://blbadger.github.io}}/logistic_map/logistic_zoom4.png)

In other words, let's call the set favored values, $\mathbf A$, obtained for any value of $r$ while iterating (1) to be

$$\mathbf A = \{x_1, x_2, ..., x_i\}$$ 

where $i$ is the index of the favored, most probable value. 

For any pair of elements $(x_1, x_2) \in \mathbf A$, if $x_1 - x_2 \to 0$ and $d/frac{dr} (x_1 - x_2) \to 0$ as $r$ increases, then succsessively larger values of $r$ move from aperiodicity to periodicity.

It appears that tangentially intersecting and crossing of favored values are mutally exclusive, such that for any region of $r$ there is either tangential or nontangential approach for all pairs of elements $(x_1, x_2) \in \mathbf A$. 

The idea that tangential approaching of two 'favored' populations implying soon-to-be periodicity is conjectural at this point.


### A logistic map from the Henon attractor

The Henon map equation describes a simplified Poincare section of the Lorenz attractor:

$$x_{n+1} = 1-ax_n^2 + y \\
y_{n+1} = bx_n \tag{2}$$

When

$$a = 1.4 \\
b = 0.3$$

The following map is produced:
![map]({{https://blbadger.github.io}}/logistic_map/henon_map.png)

This system (2) is discrete, but may be iterated using Euler's method as if we wanted to approximate a continuous equation:
$$
\cfrac{dx}{dt} = 1-ax^2 + y \\
\cfrac{dy}{dt} = bx_n\\
x_{n+1} \approx x_n + \cfrac{dx}{dt} \cdot \Delta t \\
y_{n+1} \approx y_n + \cfrac{dy}{dt} \cdot \Delta t 
\tag{3}
$$

With larger-than-accurate values of $\Delta t$, we have a not-quite-continuous map that can be made as follows:

```python
# import third-party libraries
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('dark_background')

def henon_attractor(x, y, a=.1, b=0.03):
	'''Computes the next step in the Henon 
	map for arguments x, y with kwargs a and
	b as constants.
	'''
	dx = 1 - a * x ** 2 + y
	dy = b * x
	return dx, dy
	
# number of iterations and step size
steps = 5000000
delta_t = 0.0047

X = np.zeros(steps + 1)
Y = np.zeros(steps + 1)

# starting point
X[0], Y[0] = 1, 1

# compute using Euler's formula
for i in range(steps):
	x_dot, y_dot = henon_attractor(X[i], Y[i])
	X[i+1] = X[i] + x_dot * delta_t
	Y[i+1] = Y[i] + y_dot * delta_t

# display plot
plt.plot(X, Y, ',', color='white', alpha = 0.1, markersize=0.1)
plt.axis('on')
plt.show()
```

If iterate (3) with $a=0.1, b = 0.03, \Delta t = 0.047 $, the following map is produced:
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/henon_logistic.jpg)

It looks like the bifurcation diagram from the logistic attractor! Closer inspection on the chaotic portion reveals an inverted Logistic-like map.
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/henon_logistic_zoom.png)

As this system is being iterated semicontinuously, we can observe the vectorfield behind the motion of the points:
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/henon_logistic_quiver2.png)

Subsequent iterations after the first bifurcation lead to the point bouncing from left portion to right portion in a stable period.  In the region of chaotic motion of the point, the vectors are ordered.
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/henon_logistic_quiver_zoom2.png)

Why is this?  The henon map has one nonlinearity: an $x^2$.  Nonlinear maps may transition from order (with finite periodicity) to chaos (a period of infinity). The transition from order to chaos for many systems occurs via period doubling leading to infinite periodicity in finite time, resulting in a logistic-like map.








