## Pendulum phase map

Imagine a pendulum swinging back and forth. We can plot the position of its tip on the x-axis and the velocity of the tip on the y-axis.  This xy plane is now called a phase space, and although it does not correspond to physical space it does tell us interesting information about the system it represents.  An excellent summary of modeling differential equations by 3B1B may be found [here](https://www.youtube.com/watch?v=p_di4Zn4wz4). 

By setting up a pendulum to obey Newton's laws, we can model how the pendulum will swing using Euler's formula to model the trajectory through phase space of the differential equations governing pendulum motion as it is slowed by friction:

```python
dx = y
dy = - a * y - b * sin(x)
```
Where the constant **a** denotes friction and the constant **b** represents the constant of gravity divided by the length of the pendulum.  It is helpful to view the vector plot for this differential system to get an idea of where a point moves at any given (x,y) coordinate

![pendulum vectors]({{https://blbadger.github.io}}pendulum_map/pendulum_vectors.png)

Imagine a ball rolling around on a plane that is directed by the vectors above. We can calculate this rolling using Euler's formula, 

```python
x_next = x_current + delta_t * dx
y_next = y_current + delta_t * dy
```

If delta_t (hereafter *dt*) is small (0.01 in this case), the following map is produced:
![pendulum image]({{https://blbadger.github.io}}pendulum_map/continuous_pendulum.png)

Now note that we can achieve a similar map with a linear differential system

```python
dx = -a * y
dy = -b * y + x 
```

which at *dt* = 0.1 yeilds

![swirl image]({{https://blbadger.github.io}}pendulum_map/linear_swirl.png)

In either case, the trajectory heads asymptotically towards the origin.  This is also true for any initial point in the vicinity of the origin, making the point (0,0) an **attractor** of the system.  As the attractor is a point, it is a 0-dimensional attractor or point attractor.


### Increasing timestep size leads to an increase in attractor dimension

Now let's increase *dt* little by little.  At *dt* = 0.02 the map looks similar to the one above just with more space betwen each point on the spiral.  This makes sense, as an increase in timestep size would lead to more motion between iterations provided a particle is in motion.

![pendulum image]({{https://blbadger.github.io}}pendulum_map/pendulum_0.2t.png)


Increasing *dt* to 0.037 leads to the appearance of ripples in the trajectory path, where the ratio between the distance between consecutive iteration (x, y) coordinates compared to the (x, y) coordinates of the next nearest neighbor changes depending on where in the trajectory the particle is.  For lack of a better word, let's call these **waves**.  Another way to think about these waves is to see that they are locations of apparent changes in spiral direction.

![pendulum image]({{https://blbadger.github.io}}pendulum_map/pendulum_0.37t.png)


With a slightly larger *dt* (0.04088), the waves have become more pronounced and an empty space appears around the origin (picture is zoomed slightly).

![pendulum image]({{https://blbadger.github.io}}pendulum_map/pendulum_0.04088t.png)


And by dt = 0.045, the attractor is now a ring

![pendulum image]({{https://blbadger.github.io}}pendulum_map/pendulum_0.045t.png)


Thus an increase in *dt* leads to the transformation of the pendulum map from a 0-dimensional attractor to a 1-dimensional one. Further increases in *dt* leads to explosion towards infinity.

What happens to the linear spiral system when *dt* increases? At *dt* = 0.5, the points along the spiral are slightly more spaced out

![spiral image]({{https://blbadger.github.io}}pendulum_map/spiral_map_0.5t.png)

When *dt* = 0.9, there is less space between (x, y) coordinates of different rotations than of consecutive iterations:

![spiral image]({{https://blbadger.github.io}}pendulum_map/spiral_map_0.9t.png)

And when *dt* = 0.9999, this effect is so pronounced that there appears to be a ring attractor,

![spiral image]({{https://blbadger.github.io}}pendulum_map/spiral_map_0.9999t.png)

But this is not so!  Closer inspection of this ring reveals that there is no change in point density between the starting and ending ring: instead, meaning that the points are still moving towards the origin at a constant rate.

![spiral image]({{https://blbadger.github.io}}pendulum_map/spiral_map_zoom.png)


Only at *dt* = 1 is there a 1-dimensional , but this is unstable: at small values less than or greater than 1, iterations head towards the origin or else towards infinity. 

###  Pendulum maps with 1-dimensional attractors have fractal wave patterns

Take *dt* to be 0.04087, which produces a similar map to that found above for *dt* = 0.04088 .  Now let's zoom in on the upper part of the map:

![pendulum image]({{https://blbadger.github.io}}pendulum_map/pendulum_0.04087t_zoom1.png)
![pendulum image]({{https://blbadger.github.io}}pendulum_map/pendulum_0.04087t_zoom2.png)
![pendulum image]({{https://blbadger.github.io}}pendulum_map/pendulum_0.04087t_zoom3.png)
![pendulum image]({{https://blbadger.github.io}}pendulum_map/pendulum_0.04087t_zoom4.png)

Notice that more and more waves are visible as the scale decreases: the wave pattern is a fractal.  

Where do these waves come from? They are not observed for the linear map at any *dt* size (here at 0.9999):
![pendulum image]({{https://blbadger.github.io}}pendulum_map/swirl_map_zoom.png)

and therefore are not an inevitable result of increasing *dt* for any spiral system.  Instead, they are the result of irrational numbers (or at least finiate approximations of irrationals) generated by repeated iterations of applying the trigonometric sine function to rational numbers. The same waves (more spaced out, better seen as changes in apparent winding direction) are seen when integers are plotted on the polar coordinates (see [this](https://www.youtube.com/watch?v=EK32jo7i5LQ) video for more). 

The appearance of a change in attractor dimension, self-similar fractal patterns, and irrational numbers is no coincidence, as there are deep connections between these ideas.


### Reduction of a Clifford system to the pendulum map

There are a number of similarities between widely different nonlinear systems.  Perhaps the most dramatic example of this is the ubiquitous appearance of self-similar fractals in chaotic nonlinear systems (as seen above).  This may be most dramatically seen when the constant parameters of certain equation systems are tweaked such that the output produces a near-copy of another equation system, a phenomenon that is surprisingly common to nonlinear systems. For example, take the Clifford attractor:

```python
x_dot = sin(a*y) + c*cos(a*x) 
y_dot = sin(b*x) + d*cos(b*y)
```

This is clearly and very different equation system than one modeling pendulum swinging, and for most constant values it produces a variety of maps that look nothing like what is produced by the pendulum system.  But observe what happens when we iterate semicontinuously, setting

```python
a=-0.3, b=0.2, c=0.5, d=0.3, delta_t = 0.9
(x[0], y[0]) = (90, 90)
```

We have a (slightly oblong) pendulum map!

![clifford pendulum image]({{https://blbadger.github.io}}pendulum_map/clifford_pendulum.png)


