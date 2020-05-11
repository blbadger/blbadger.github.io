## Pendulum phase map

Imagine a pendulum swinging back and forth. We can plot the position of its tip on the x-axis and the velocity of the tip on the y-axis.  This xy plane is now called a phase space, and although it does not correspond to physical space it does tell us interesting information about the system it represents.  An excellent summary of modeling differential equations by 3B1B may be found [here](https://www.youtube.com/watch?v=p_di4Zn4wz4)

By setting up a pendulum to obey Newton's laws, we can model how the pendulum will swing using Euler's formula to model the trajectory through phase space of the differential equations governing pendulum motion:








The following images are interesting examples of what appears to be floating-point computations inaccuracies.  If we zoom far in on the center of a semicontinuous pendulum map with just-larger-than-accurate t sizes, we can find interesting accretion-disk like patterns.  These patterns do not form if the pendulum phase map spiral is centered on the origin where errors proportional to the size of the numbers being computed remains small at arbitrary scale.  Node the X-offset in all images corresponding to the off-origin center of these maps.


Being the results of 
![pendulum image]({{https://blbadger.github.io}}pendulum_map/pendulum_accretion.png)

If we shift the value of G by a factor of one trillionth, a very different pattern appears:
![pendulum image]({{https://blbadger.github.io}}pendulum_map/pendulum_accretion_g-shifted.png)

If we then take this new G value and shift the starting X value by a factor of one trillionth, again a new pattern is seen:
![pendulum image]({{https://blbadger.github.io}}pendulum_map/pendulum_shifted_g_and_y.png)

Linear systems can also produce these patterns:
![linear pendulum image]({{https://blbadger.github.io}}pendulum_map/linear_swirl_accretion.png)



