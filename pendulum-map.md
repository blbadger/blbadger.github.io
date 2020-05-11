## Pendulum phase map

Imagine a pendulum swinging back and forth. We can plot the position of its tip on the x-axis and the velocity of the tip on the y-axis.  This xy plane is now called a phase space, and although it does not correspond to physical space it does tell us interesting information about the system it represents.  An excellent summary of modeling differential equations by 3B1B may be found [here](https://www.youtube.com/watch?v=p_di4Zn4wz4)

By setting up a pendulum to obey Newton's laws, we can model how the pendulum will swing using Euler's formula to model the trajectory through phase space of the differential equations governing pendulum motion as it is slowed by friction:

'''
dx = y
dy = - a * y - b * sin(x)
'''

Where the constant 'a' denotes friction and the constant 'b' represents the constant of gravity divided by the lenght of the pendulum.

## Nonlinearity leads to a fractal pattern and a change in dimension


## Reduction of a Clifford system to the pendulum map, iterated semicontinuously

There are a number of deep similarities between widely different nonlinear systems.  Perhaps the most dramatic example of this is the ubiquitous appearance of self-similar fractals in chaotic nonlinear systems (as seen above).  This may be most dramatically seen when the constant parameters of certain equation systems are tweaked such that the output produces a near-copy of another equation system, a phenomenon that is surprisingly common to nonlinear systems. For example, take the Clifford attractor:

'''
x_dot = np.sin(a*y) + c*np.cos(a*x) 
y_dot = np.sin(b*x) + d*np.cos(b*y)
'''

This is clearly and very different equation system than one modeling pendulum swinging, and for most constant values it produces a variety of maps that look nothing like what is produced by the pendulum system.  But observe what happens when we iterate semicontinuously, setting

'''
a=-0.3, b=0.2, c=0.5, d=0.3, delta_t = 0.9
(x[0], y[0]) = (90, 90)
'''

We have a (slightly oblong) pendulum map!

![clifford pendulum image]({{https://blbadger.github.io}}pendulum_map/clifford_pendulum.png)


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following images are interesting examples of what appear to be floating-point computation inaccuracies.  If we zoom far in on the center of a semicontinuous pendulum map with just-larger-than-accurate t sizes, we can find interesting accretion-disk like patterns.  These patterns do not form if the pendulum phase map spiral is centered on the origin where errors proportional to the size of the numbers being computed remains small at arbitrary scale.  Node the X-offset in all images corresponding to the off-origin center of these maps.

These particular computational artefacts are interresting because they demonstrate sensitivity to initial conditions in a deterministic system that is characteristic of chaotic systems. 

For a certain starting system:
![pendulum image]({{https://blbadger.github.io}}pendulum_map/pendulum_accretion.png)

If we shift the value of G by a factor of one trillionth, a very different pattern appears:
![pendulum image]({{https://blbadger.github.io}}pendulum_map/pendulum_accretion_g-shifted.png)

If we then take this new G value and shift the starting X value by a factor of one trillionth, again a new pattern is seen:
![pendulum image]({{https://blbadger.github.io}}pendulum_map/pendulum_shifted_g_and_y.png)

Linear systems can also produce these patterns:
![linear pendulum image]({{https://blbadger.github.io}}pendulum_map/linear_swirl_accretion.png)



