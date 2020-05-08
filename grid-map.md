## Sine-Cosine grid map

### A demonstration of semicontinuous mapping of a 2D chaotic system

The differential system:

x_next = x_current + 0.1 * cos(y_current) * delta_t

y_next = y_current + 0.1 * sin(x_current) * delta_t
  
  
The vector map
![t=0.05 map]({{https://blbadger.github.io}}/grid_map/cossin_vectors.png)


Observe what happens with increases in the step size:

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
