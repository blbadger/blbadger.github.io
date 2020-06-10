## The Henon map

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


### A semicontinuous iteration of the Henon map reveals period doubling 

If iterate (3) with $a=0.1, b = 0.03, \Delta t = 0.047 $, the following map is produced:
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/henon_logistic.jpg)

It looks like the bifurcation diagram from the logistic attractor! Closer inspection on the chaotic portion reveals an inverted Logistic-like map.
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/henon_logistic_zoom.png)

As this system is being iterated semicontinuously, we can observe the vectorfield behind the motion of the points:
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/henon_logistic_quiver2.png)

Subsequent iterations after the first bifurcation lead to the point bouncing from left portion to right portion in a stable period.  In the region of chaotic motion of the point, the vectors are ordered.
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/henon_logistic_quiver_zoom2.png)

Why is this?  The henon map has one nonlinearity: an $x^2$.  Nonlinear maps may transition from order (with finite periodicity) to chaos (a period of infinity). The transition from order to chaos for many systems occurs via period doubling leading to infinite periodicity in finite time, resulting in a logistic-like map.

