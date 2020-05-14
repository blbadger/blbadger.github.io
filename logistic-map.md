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

A fundamental feature of aperiodicity is extreme sensitivity to initial conditions, such that long-range behavior is impossible to predict.  Observe what happens when the starting population proportion is shifted by a factor of one ten-millionth:
![map]({{https://blbadger.github.io}}/logistic_map/logistic_time_r3.7_comp.png)

The behavior is identical to the unshifted population for a while, until it changes and becomes wildly different.  This sensitivity to initial conditions was found by the pioneering Lorenz while studying convection.  Necessarily imperfect initial measurements prevent long-range prediction.


### The relationship between chaotic systems and fractals

This information may be compiled in what is called an orbit map, which displays the stable points at each value of r.  These may also be though of as the roots of the equation with specific r values. 

On the x-axis is r, on the y-axis is the stable point value. By looking at how many points there are at a given r value, the same patter of period doubling may be observed. The phenomenon that periodic nonlinear systems become aperiodic via period doubling at specific ratios was found by Feigenbaum to be a near-universal feature of the transition from periodicity to chaos.


![map]({{https://blbadger.github.io}}/logistic_map/logistic_period.png)

Let's take a closer look at the fuzzy region of the right. This corresponds to the values of $r$ which are mostly aperiodic, but with windows of periodicity.  There are all kinds of interesting shapes visible, highlighting a key difference between mathematical chaos and the normal English word (OED: a state of complete confusion and lack of order). 
![map]({{https://blbadger.github.io}}/logistic_map/logistic_period_zoom2.png)

One of the most striking features of this map is that it is a self-similar fractal.  This means that smaller parts resemble the whole object.  Observe what happens when we zoom in on the upper left section: A smaller copy of the original image is found.
![map]({{https://blbadger.github.io}}/logistic_map/logistic_period_zoom3.png)

If we take this image and again zoom in on the upper left hand corner, again we see the original!
![map]({{https://blbadger.github.io}}/logistic_map/logistic_period_zoom4.png)

And so on ad infinitum.  An infinite number of smaller images of the original are found.  Far from being unusual, the formation of fractals from nonlinear systems is the norm provided that they are dissipative and do not explode towards infinity.


### A logistic map from the Henon attractor

The Henon map equation describes a simplified Poincare section of the Lorenz attractor:

$$x_{n+1} = 1-ax_n^2 + y \\
y_{n+1} = bx_n \tag{2}$$

When

$$a = 1.4 \\
b = 0.3$$

The following map is produced:
![map]({{https://blbadger.github.io}}/logistic_map/henon_map.png)

This system is discrete, but may be iterated using Euler's method (see Clifford attractor page for details). With larger-than-accurate values of $\Delta t$, we have a not-quite-continuous map that can be made as follows:

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

If iterated using Euler's formula with $\Delta t = 0.047 $, the following map is produced:
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/henon_logistic.jpg)

It looks like the bifurcation diagram from the logistic attractor! Closer inspection on the chaotic portion reveals an inverted Logistic-like map.
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/henon_logistic_zoom.png)

As this system is being iterated semicontinuously, we can observe the vectorfield behind the motion of the points:
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/henon_logistic_quiver2.png)

Subsequent iterations after the first bifurcation lead to the point bouncing from left portion to right portion in a stable period.  In the region of chaotic motion of the point, the vectors are ordered.
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/henon_logistic_quiver_zoom2.png)

Why is this?  The henon map has one nonlinearity: an $x^2$.  Nonlinear maps may transition from order (with finite periodicity) to chaos (a period of infinity). The transition from order to chaos for many systems occurs via period doubling leading to infinite periodicity in finite time, resulting in a logistic-like map.








