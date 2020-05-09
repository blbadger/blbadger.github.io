## Sine-Cosine grid map

### A demonstration of semicontinuous mapping of a 2D chaotic system


The differential system:

x_next = x_current + 0.1 * cos(y_current) * delta_t

y_next = y_current + 0.1 * sin(x_current) * delta_t
  
  
The vector map
![t=0.05 map]({{https://blbadger.github.io}}/grid_map/cossin_vectors.png)


Chaotic mathematical systems are deterministic but deeply unpredictable: small changes to the starting values of a chaotic system will lead to large changes in the output. The equation system above is chaotic for a large enough delta_t.  For example, take delta_t to be 0.8 and the starting x, y coordinates to be 1, 0. The following map is produced:

![t=0.8 map]({{https://blbadger.github.io}}/grid_map/cossin_0.8t.png)

if the starting x coordinate is shifted by a factor of one billionth (to 1.000000001), a completely different map is produced:

![t=0.5 shifted map]({{https://blbadger.github.io}}/grid_map/cossin_0.8t_shifted.png)


Euler's formula has been used to estimate the trajectory of unsolvable differential equaitons.  Here it is used with dliberately large values of delta_t in order to demonstrate a mapping that changes from a continuous trajectory to a discrete-like map. Observe what happens with increases in the step size:

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
