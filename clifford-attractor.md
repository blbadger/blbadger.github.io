## Semicontinuous Clifford attractor

The clifford attractor, also known as the fractal dream attractor, is the system of equations:

$$
x_{n+1} = sin(ay_n) + c \cdot cos(ax_n) \\
y_{n+1} = sin(bx_n) + d \cdot cos(by_n)
\tag{1}$$


where a, b, c, and d are constants of choice.  It is an attractor because at given values of a, b, c, and d,
any starting point $(x_0, y_0)$ will wind up in the same pattern. See Vedran Sekara's post [here](https://vedransekara.github.io/2016/11/14/strange_attractors.html) for a good summary on how to use Python to make a plot of the Clifford attractor.

with $a = 2, b = 2, c = 1, d = -1$ the following map is made:
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_1.png)

with $a = 2, b = 1, c = -0.5, d = -1.01$ , 
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_2.png)

A wide array of shapes may be made by changing the constants, and the code to do this is 
[here](https://github.com/blbadger/2D_strange_attractors/blob/master/clifford_attractor.py)
Try experimenting with different constants!

### Clifford attractors are fractal

In what way is this a fractal?  Let's zoom in on the lower right side: 
In the central part that looks like Saturn's rings, there appear to be 6 lines.
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_zoom1.png)

At a smaller scale, however, there are more visible, around 10
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_zoom2.png)

Is this all? Zooming in once, I count 14 lines, with 7 on the top section
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_zoom3.png)

Zooming in on the top shows that there are actually more that 14 lines!
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_zoom4.png)

To drive the point home, the top-right line looks to be as solid a line as any, but on closer inspection it is not. 
There are two paths visible at higher magnification:
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_zoom5.png)

### Clifford attractors can shift between one and two (or more) dimensions

For $a = 2.1, b = 0.8, c = -0.5, d = -1$, the attractor is three points:
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_0d.png)

For $b = 0.95$, more points:
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_0d2.png)

For $b = 0.981$, the points being to connect to form line segments
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_1d1.png)

And for $b = 0.9818$, a slightly-larger-than 1D attractor is produced
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_1d2.png)

And when $b = 1.7$, a nearly-2d attractor is produced
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_2d.png)

### Semi-continuous mapping

Say you want to model a continuous ordinary differential equation.  If the equation is nonlinear, chances are that there is no analytic solution.  What is one to do? Do an approximation! Perhaps the simplest way of doing this is by using discrete approximations to estimate where a point will go given its current position and its derivative.  This is known as Euler's method, and can be expressed as follows:
If 

$$
dx / dt = f(x), \\
x(0) = C
$$

then 

$$
x_{next} \approx x_{current} + dx/dt \cdot \Delta t
$$

With smaller and smaller values of $\Delta t$, the approximation becomes better and better but more and more computations are required for the same desired time interval:

$$
x_{next} = x_{current} + dx/dt \Delta_t \quad as \, \Delta_t \to 0
$$

For a two dimensional equation, the approximations can be made in each dimension:

$$
x_{n+1} \approx x_n + dx \cdot \Delta t \\
y_{n+1} \approx y_n + dy \cdot \Delta t
$$

To make these calculations and plotting them in python, the wonderful numpy and matplotlib libraries are used and we define the Clifford attractor function:
```python
# import third party libraries
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('dark_background')

def clifford_attractor(x, y, a=-1.4, b=1.7, c=1.0, d=0.7):
	'''Returns the change in arguments x and y according to 
	the Clifford map equation. Kwargs a, b, c, and d are specified
	as constants.
	'''
	x_next = np.sin(a*y) + c*np.cos(a*x) 
	y_next = np.sin(b*x) + d*np.cos(b*y)
	return x_next, y_next
```
Setting up the number of iterations and the time step size, we then initialize the numpy array with 0s and add a starting $x, y$ coordinate:

```python
# number of iterations
iterations = 1000000
delta_t = 0.01

# initialization
X = np.zeros(iterations)
Y = np.zeros(iterations)

# starting point
(X[0], Y[0]) = (10.75, 8.2)
```

For computing Euler's formula let's loop over the clifford function, adding in each next computed value to the numpy array.
```python 
# euler's method for tracking differential equations
for i in range(iterations-1):
	x_next, y_next = clifford_attractor(X[i], Y[i])
	X[i+1] = X[i] + x_next * delta_t
	Y[i+1] = Y[i] + y_next * delta_t
```

Now let's plot the graph! 
```python
# make and display figure
plt.figure(figsize=(10, 10))

# differential trajectory
plt.plot(X, Y, ',', color='white', alpha = 0.2, markersize = 0.05)
plt.axis('on')
plt.show()
```

at $\Delta t = 0.01$, a smooth path along the vectors is made.  The path is 1D, and the attractor is a point (0D).
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_0.01t.png)

at $\Delta t = 0.1$
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_0.1t.png)

at $\Delta t = 0.8$ (points are connected for clarity)
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_0.8t.png)

$\Delta t = 1.1$
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_1.1t.png)

$\Delta t = 1.15$
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_1.15t.png)

$\Delta t = 1.2$, the first few iterations reveal four slowly rotating lattice points
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_1.2t_lines.png)

with more iterations at $\Delta t$ = 1.2, it is clear that the attractor is now 1 dimensional
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_1.2t.png)

$\Delta t = 1.3$
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_1.3t.png)

$\Delta t = 1.35$
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_1.35t_lines.png)

$\Delta t = 1.35$, a shape similar to the discrete map has formed.
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_1.35t.png)

### Is this a fractal? 

Zooming in on the bottom right section suggests that it is:
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_zoom1.png)
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_zoom2.png)
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_zoom3.png)

### In what way is this mapping semi-continuous?







