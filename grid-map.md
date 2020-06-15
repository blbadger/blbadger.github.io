## Sine-Cosine grid map

### A demonstration of semicontinuous mapping of a 2D chaotic system

The differential system
$$ \cfrac{dx}{dt} = a \cdot cos(y) \\
\cfrac{dy}{dt} = b \cdot sin(x) \tag{1} $$

The vector map of this equation is as follows:
![t=0.05 map]({{https://blbadger.github.io}}/grid_map/cossin_vectors.png)

To evaluate this equation with Euler's method:

$$
x_{n+1} = x_n + \cfrac{dx}{dt} \Delta t \\
y_{n+1} = y_n + \cfrac{dy}{dt} \Delta t  \tag{2}
$$

Chaotic mathematical systems are deterministic but deeply unpredictable: small changes to the starting values of a chaotic system will lead to large changes in the output. The equation system above is chaotic for a large enough $\Delta t$.  For example, take $\Delta t = 0.8$ and the starting $x, y$ coordinates to be $1, 0$. The following map is produced:

![t=0.8 map]({{https://blbadger.github.io}}/grid_map/cossin_0.8t.png)

If the starting $x$ coordinate is shifted by a factor of one billionth (to 1.000000001), a completely different map is produced:

![t=0.5 shifted map]({{https://blbadger.github.io}}/grid_map/cossin_0.8t_shifted.png)

Animating the trajectory of both of these maps with $x_{01} = 1$ in red and $x_{02} = 1.000000001$ in blue, we have 

![t=0.5 shifted map]({{https://blbadger.github.io}}/grid_map/grid_vid.gif)


Euler's formula is used to (not very accurately) estimate the trajectory of unsolvable differential equations.  Here it is employed with deliberately large values of delta_t in order to demonstrate a mapping that is not quite continuous but not a classic recurrence (discrete) mapping either.

This idea becomes clearer when the vector map is added to the trajectory.  Observe how the particles are influenced by the vectors, as is the case for a continuous trajectory, 

![t=0.05 map]({{https://blbadger.github.io}}/grid_map/cossin_quivers.png)

and that on close inspection there are gaps between successive iterations, as for a discrete recursive map
![t=0.05 map]({{https://blbadger.github.io}}/grid_map/cossin_quivers_zoom.png)

Systems of ordinary differential equations have one independent variable: time.  This leads to all trajectories being unique.  This being the case, how is this map possible given that each vertex point seems to lead to multiple trajectories?  The answer is that the trajectories get very close to each other but do not touch.  At a smaller scale, an intersection of grids is revealed to not be an intersection at all.

![t=0.05 map]({{https://blbadger.github.io}}/grid_map/grid_map_intersection.png)

### The grid map with larger $\Delta t$ values is indistinguisheable from a random walk browninan trajectory

Imagine a ball with elastic collisions to sparse particles that flow in the vector map pattern, or else a ball moving smoothly that is only influenced by the vectors at discrete time intervals. Observe what happens with increases in the time step size:

$\Delta t = 0.05$
![t=0.05 map]({{https://blbadger.github.io}}/grid_map/cossin_0.05t.png)

$\Delta t = 0.5$
![t=0.5 map]({{https://blbadger.github.io}}/grid_map/cossin_0.5t.png)

$\Delta t = 13$
![t=13 map]({{https://blbadger.github.io}}/grid_map/cossin_13t.png)

$\Delta t = 15$
![t=15 map]({{https://blbadger.github.io}}/grid_map/cossin_15t.png)

$\Delta t = 18$
![t=18 map]({{https://blbadger.github.io}}/grid_map/cossin_18t.png)

Which has a trajectory that is formed as follows
![t=18 map]({{https://blbadger.github.io}}/grid_map/grid_18.gif)

and still remains extremely sensitive to inital values ($x_0 = 1$ in red, $x_0 = 1.000000001$ in blue).

![t=18 map]({{https://blbadger.github.io}}/grid_map/grid_18_comp.png)


With increases in $\Delta t$, the map's fractal dimension increases. It is not impossible for 2-dimensional continuous differential equations to produce a strange (fractal) attractor, but it is possible for a 2D discrete system to do so.

At $\Delta t = 18$, the trajectory is indistinguisheable from random walk, which is often modelled mathematically by a system called a ([Weiner process](https://en.wikipedia.org/wiki/Brownian_motion)).  This is not peculiar to the equation system (1) but is a feature of many nonlinear systems (see the logistic attractor or Clifford attractor pages) that are iterated discontinuously.  

Why is this important?  It means that real observations that are normally ascribed to a stochastic (usually linear) model are equally ascribable deterministic nonlinear equation systems.  And this is important because once we have perfomed an inversion with respect to what can be ascribed to stochastic versus deterministic events, we can invert the reasoning on what is insignificant data ('noise') versus what is significant ('signal').  



