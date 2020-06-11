## The Henon map

Maurice Henon sought to recapitulate the geometry of the Lorenz attractor in two dimensions.  This requires stretching and folding of space

Henon investigated the following discrete system ([here](https://projecteuclid.org/euclid.cmp/1103900150)), which is now referred to as the Henon map:

$$x_{n+1} = 1-ax_n^2 + y \\
y_{n+1} = bx_n \tag{1}$$

When

$$a = 1.4 \\
b = 0.3 \\
x_0, y_0 = 0, 0
$$

Successive iterations jump around unpredictably but are attracted to a distinctive curved shape.  For the first thousand iterations:
![map]({{https://blbadger.github.io}}/henon_map/henon_dev.gif)

After many iterations, the following map is produced:
![map]({{https://blbadger.github.io}}/logistic_map/henon_map.png)


### The Henon map is a strange (fractal) attractor

For certain starting values $x_0, y_0$, (1) with a=1.4 and b=0.3 does not head towards infinity but is instead attracted to the region shown above.  This shape is called an attractor because regardless of where $x_0, y_0$ is placed, if subsequent iterations do not diverge then they are drawn to the shape above.  

Let's examine this attractor.  If we increase magnification on the top line in the center, we find that it is not a line at all!  With successive increases in magnification (and more iterations of (1)), we can see that each top line is actually many lines close together, in a self-similar pattern.  This is indicative of a fractal shape called the Cantor set.

![map]({{https://blbadger.github.io}}/henon_map/henon_zoom1.png)

![map]({{https://blbadger.github.io}}/henon_map/henon_zoom2.png)

![map]({{https://blbadger.github.io}}/henon_map/henon_zoom3.png)

![map]({{https://blbadger.github.io}}/henon_map/henon_zoom4.png)

As a video,
![map]({{https://blbadger.github.io}}/henon_map/henon_zoom.gif)

In general terms, the Henon map is a fractal because it looks similar at widely different scales.  

### The boundary of the basin of attraction for the Henon map 



### A semicontinuous iteration of the Henon map reveals period doubling 

This map (1) is discrete, but may be iterated using Euler's method as if we wanted to approximate a continuous equation:
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

Why is this?  The (1) has one nonlinearity: an $x^2$.  Nonlinear maps may transition from order (with finite periodicity) to chaos (a period of infinity). The transition from order to chaos for many systems occurs via period doubling leading to infinite periodicity in finite time, resulting in a logistic-like map.

### Pendulum map from the Henon attractor

This is not the only similarity the Henon map has to another system: (1) can also result in a map that displays the waves of the pendulum map, explained [here](/pendulum-map.md).

