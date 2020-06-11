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

This can be plotted using python as follows:

```python
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('dark_background')

def henon_attractor(x, y, a=1.4, b=0.3):
	'''Computes the next step in the Henon 
	map for arguments x, y with kwargs a and
	b as constants.
	'''
	x_next = 1 - a * x ** 2 + y
	y_next = b * x
	return x_next, y_next
	
# number of iterations and array initialization
steps = 100000
X = np.zeros(steps + 1)
Y = np.zeros(steps + 1)

# starting point
X[0], Y[0] = 0, 0

# add points to array
for i in range(steps):
	x_dot, y_dot = henon_attractor(X[i], Y[i])
	X[i+1] = x_dot 
	Y[i+1] = y_dot
	
# plot figure
plt.plot(X, Y, '^', color='white', alpha = 0.8, markersize=0.3)
plt.axis('off')
plt.show()
plt.close()
```

After many iterations, the following map is produced:
![map]({{https://blbadger.github.io}}/logistic_map/henon_map.png)

How does the equation produce the map above?  We can plot each point one by one to find out.  To do this, the program above can be modified as follows to make many images of the map of successive iterations of (1), which can then be compiled into a movie (see [here](/julia-sets.md) for an explanation on how to compile images using ffmpeg).

```python
...

for i in range(steps):
	x_dot, y_dot = henon_attractor(X[i], Y[i])
	X[i+1] = x_dot 
	Y[i+1] = y_dot 
	plt.xlim(-1.5, 1.5)
	plt.ylim(-0.5, 0.5)

	plt.plot(X, Y, '^', color='white', alpha = 0.8, markersize=0.3)
	plt.axis('off')
	plt.savefig('{}.png'.format(i), dpi=300)
	plt.close()
```

For the first thousand iterations:
![map]({{https://blbadger.github.io}}/henon_map/henon_dev.gif)

Successive iterations jump around unpredictably but are attracted to a distinctive curved shape. 

### The Henon map is a strange (fractal) attractor

For certain starting values $x_0, y_0$, (1) with a=1.4 and b=0.3 does not head towards infinity but is instead attracted to the region shown above.  This shape is called an attractor because regardless of where $x_0, y_0$ is placed, if subsequent iterations do not diverge then they are drawn to the shape above.  

Let's examine this attractor.  If we increase magnification on the top line in the center, we find that it is not a line at all!  With successive increases in magnification (and more iterations of (1)), we can see that each top line is actually many lines close together, in a self-similar pattern.  This is indicative of a fractal shape called the Cantor set.

![map]({{https://blbadger.github.io}}/henon_map/henon_zoom1.png)

![map]({{https://blbadger.github.io}}/henon_map/henon_zoom2.png)

![map]({{https://blbadger.github.io}}/henon_map/henon_zoom3.png)

![map]({{https://blbadger.github.io}}/henon_map/henon_zoom4.png)

In general terms, the Henon map is a fractal because it looks similar at widely different scales.  

### The boundary of the basin of attraction for the Henon map 

Some experimentation can convince us that not all starting points head towards the attractor upon successive iterations of (1) with $a=1.4$ and $b=0.3$: instead, some head towards positive or negative infinity!  The collection of points that do not diverge (head towards infinity) for a given dynamical system is called the basin of attraction.  Basins of attraction may be fractal or else smooth as shown by Yorke [here](http://yorke.umd.edu/Yorke_papers_most_cited_and_post2000/1985_05_McDonald_GOY_Fractal_basin_boundaries.pdf).  Does the Henon map (with $a=1.4, b=0.3$) have a smooth or fractal basin?

To find out, let's define a function

```python
# import third-party libraries
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('dark_background')
import copy

def henon_boundary(max_iterations, a, b):
	x_range = 2000
	y_range = 2000

	x_list = np.arange(-5, 5, 10/x_range)
	y_list = np.arange(5, -5, -10/y_range)
	array = np.meshgrid(x_list, y_list)

	x2 = np.zeros(x_range)
	y2 = np.zeros(y_range)
	iterations_until_divergence = np.meshgrid(x2, y2)

	for i in iterations_until_divergence:
		for j in i:
			j += max_iterations

	for k in range(max_iterations):
		array_copied = copy.deepcopy(array[0]) # copy array to prevent premature modification of x array

		# henon map applied to array 
		array[0] = 1 - a * array[0]**2 + array[1]
		array[1] = b * array_copied


		r = (array[0]**2 + array[1]**2)**0.5
		diverging = r > 10
		iterations_until_divergence[0][diverging] = k

		# prevent future divergence
		array[0][diverging] = 0
		array[1][diverging] = 0

	return iterations_until_divergence[0]
```

And now this may be plotted.

```python
plt.imshow(henon_boundary(70 + 2*t, a=0.2, b=-1.1), extent=[-2, 2, -2, 2], cmap='twilight_shifted', alpha=1)
plt.axis('off')
plt.savefig('{}.png'.format(i), dpi=300)
plt.close()
```

Let's see what happens to the basin of attraction and the attractor itself when $a$ is increased from $1$ to $1.48$ (constant  $b=0.3$):

![map]({{https://blbadger.github.io}}/henon_map/henon_boundary_1_to_1.48.gif)

The attractor is visible as long as it remains in the basin of attraction.  This intuitively makes sense: there is nothing special about the original points compared to subsequent iterations.  If points in an attractor were drawn to a region that then blew up to infinity, the attractor would be no more no matter where the starting point was located. Focusing on the transition from smooth to fractal form in the basin of attraction, we can see this coincides with the disappearence of the attractor itself:

![map]({{https://blbadger.github.io}}/henon_map/henon_boundary_1.4_to_1.49.gif)

At $a-0.2$ and $b=-1.1$, points head towards infinity nearly everywhere. But a pinwheel-like pattern is formed by the areas of slower divergence. Let's zoom in on this pinwheel to get an appreciation of its structure!  The first step is to pick a point and then adjust the array the graph is produced on accordingly.

```python
def henon_map(max_iterations, a, b, x_range, y_range):
	xl = -5/(2**(t/15)) + 0.459281
	xr = 5/(2**(t/15)) + 0.459281
	yl = 5/(2**(t/15)) -0.505541
	yr = -5/(2**(t/15)) -0.505541

	x_list = np.arange(xl, xr, (xr - xl)/x_range)
	y_list = np.arange(yl, yr, -(yl - yr)/y_range)
	array = np.meshgrid(x_list, y_list)
```
Now we can plot the images produced (adjusting the `extent` variable if the correct scale labels are desired) in a for loop.  I ran into a difficulty here that I could not completely debug: the `diverging` array in the `henon_map()` function occasionally experienced an off-by-one error in indexing: if the `array[0]` dimensions are 2000x2000, the `diverging` array would become 2001x2001 or 2002x2002 etc.  A try/except workaround is shown below that is quick and effective but does not really solve the fundamental problem.

```python
x_range = 2005
y_range = 2005
for t in range(300):
	# There is a strange off-by-one error in making the diverging array (such that it is one larger than it should be)
	# To address this, the original array is enlarged until the index error no longer occurs
	while True:
		end = True
		try: plt.imshow(henon_map(70 + t, a=0.2, b=-1.1, x_range=x_range, y_range = y_range), extent=[-2/(2**(t/15)) + 0.459281, 2/(2**(t/15))+ 0.459281, -2/(2**(t/15))-0.505541,2/(2**(t/15)) -0.505541], cmap='twilight_shifted', alpha=1)
		except IndexError:
			x_range += 1
			y_range += 1
			end = False
		if end: break

	plt.axis('off')
	plt.savefig('{}.png'.format(t), dpi=300)
	plt.close()
```

When $a=0.2, b=1.1$, increasing the scale by a factor of $2^7$ around the point $(x, y) = (0.459281, -0.505541) we have
![map]({{https://blbadger.github.io}}/henon_map/henon_boundary_zoom.gif)

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



