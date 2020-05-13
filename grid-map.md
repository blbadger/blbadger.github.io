## Sine-Cosine grid map

### A demonstration of semicontinuous mapping of a 2D chaotic system

The differential system:
```python
dx = 0.1 * cos(y)
dy = 0.1 * sin(x) 
```

The vector map of this equation is as follows:
![t=0.05 map]({{https://blbadger.github.io}}/grid_map/cossin_vectors.png)


To evaluate this equation with Euler's method:

```python
x_next = x_current + delta_t * dx
y_next = y_current + delta_t * dy
```

Chaotic mathematical systems are deterministic but deeply unpredictable: small changes to the starting values of a chaotic system will lead to large changes in the output. The equation system above is chaotic for a large enough delta_t.  For example, take delta_t to be 0.8 and the starting x, y coordinates to be 1, 0. The following map is produced:

![t=0.8 map]({{https://blbadger.github.io}}/grid_map/cossin_0.8t.png)

If the starting x coordinate is shifted by a factor of one billionth (to 1.000000001), a completely different map is produced:

![t=0.5 shifted map]({{https://blbadger.github.io}}/grid_map/cossin_0.8t_shifted.png)


Euler's formula has been used to estimate the trajectory of unsolvable differential equations.  Here it is employed with deliberately large values of delta_t in order to demonstrate a mapping that is not quite continuous but not a classic recurrence (discrete) mapping either.

This idea becomes clearer when the vector map is added to the trajectory.  Observe how the particles are influenced by the vectors, as is the case for a continuous trajectory, 

![t=0.05 map]({{https://blbadger.github.io}}/grid_map/cossin_quivers.png)

and that on close inspection there are gaps between successive iterations, like a discrete recursive map
![t=0.05 map]({{https://blbadger.github.io}}/grid_map/cossin_quivers_zoom.png)


Imagine a ball with elastic collisions to sparse particles that flow in the vector map pattern, or else a ball moving smoothly that is only influenced by the vectors at discrete time intervals. Observe what happens with increases in the time step size:

delta_t = 0.05
![t=0.05 map]({{https://blbadger.github.io}}/grid_map/cossin_0.05t.png)

delta_t = 0.5
![t=0.5 map]({{https://blbadger.github.io}}/grid_map/cossin_0.5t.png)

delta_t = 13
![t=13 map]({{https://blbadger.github.io}}/grid_map/cossin_13t.png)

delta_t = 15
![t=15 map]({{https://blbadger.github.io}}/grid_map/cossin_15t.png)

delta_t = 18
![t=18 map]({{https://blbadger.github.io}}/grid_map/cossin_18t.png)

With increases in delta_t, the map's fractal dimension increases. It is not impossible for 2-dimensional continuous differential equations to produce a strange (fractal) attractor, but it is possible for a 2D discrete system to do so.
