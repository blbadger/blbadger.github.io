## Pendulum map

Imagine a pendulum swinging back and forth. We can plot the position of its tip on the x-axis and the velocity of the tip on the y-axis.  This xy plane is now called a phase space, and although it does not correspond to physical space it does tell us interesting information about the system it represents.  An excellent summary of modeling differential equations by 3B1B may be found [here](https://www.youtube.com/watch?v=p_di4Zn4wz4). 

By setting up a pendulum to obey Newton's laws, we can model how the pendulum will swing using Euler's formula to model the trajectory through phase space of the differential equations governing pendulum motion as it is slowed by friction:

$$
dx = y \\
dy = -ay - b \cdot sin(x) 
\tag{1} $$

Where the constant $a$ denotes friction and the constant $b$ represents the constant of gravity divided by the length of the pendulum.  THis system of equations is nonlinear (due to the sine term) and dissipative (from the friction, $-ay$) which means that it takes a 2D area of starting points down to a 0 area.  


It is helpful to view the vector plot for this differential system to get an idea of where a point moves at any given (x,y) coordinate:

![pendulum vectors]({{https://blbadger.github.io}}pendulum_map/pendulum_vectors.png)

Imagine a ball rolling around on a plane that is directed by the vectors above. We can calculate this rolling using Euler's formula (see [here](https://blbadger.github.io/clifford-attractor.html)) the change in time step $\Delta t$ is small (0.01 in this case), the following map is produced:
![pendulum image]({{https://blbadger.github.io}}pendulum_map/continuous_pendulum.png)

Now note that we can achieve a similar map with a linear dissipative differential system
$$
dx = -ay \\
dy = -by + x \tag{2}
$$

which at $\Delta t = 0.1 $ yeilds

![swirl image]({{https://blbadger.github.io}}pendulum_map/linear_swirl.png)

In either case, the trajectory heads asymptotically towards the origin.  This is also true for any initial point in the vicinity of the origin, making the point (0,0) an **attractor** of the system.  As the attractor is a point, it is a 0-dimensional attractor or point attractor.


### Increasing timestep size leads to an increase in attractor dimension

Now let's increase $\Delta t$ little by little.  At $\Delta t = 0.02$ the map looks similar to the one above just with more space betwen each point on the spiral.  This makes sense, as an increase in timestep size would lead to more motion between iterations provided a particle is in motion.

![pendulum image]({{https://blbadger.github.io}}pendulum_map/pendulum_0.2t.png)


Increasing $\Delta t$ to 0.037 leads to the appearance of ripples in the trajectory path, where the ratio between the distance between consecutive iteration (x, y) coordinates compared to the (x, y) coordinates of the next nearest neighbor changes depending on where in the trajectory the particle is.  For lack of a better word, let's call these waves.  Another way to think about the waves is to see that they are locations of apparent changes in spiral direction.

![pendulum image]({{https://blbadger.github.io}}pendulum_map/pendulum_0.37t.png)


With a slightly larger $\Delta t$ (0.04088), the waves have become more pronounced and an empty space appears around the origin (picture is zoomed slightly).

![pendulum image]({{https://blbadger.github.io}}pendulum_map/pendulum_0.04088t.png)


And by $\Delta t = 0.045$, the attractor is now a ring

![pendulum image]({{https://blbadger.github.io}}pendulum_map/pendulum_0.045t.png)


Thus an increase in $\Delta t$ leads to the transformation of the pendulum map from a 0-dimensional attractor to a 1-dimensional one. Further increases in $\Delta t$ leads to explosion towards infinity.

What happens to the linear spiral system when $\Delta t$ increases? At $\Delta t = 0.5$, the points along the spiral are slightly more spaced out

![spiral image]({{https://blbadger.github.io}}pendulum_map/spiral_map_0.5t.png)


When $\Delta t = 0.9$, there is less space between (x, y) coordinates of different rotations than of consecutive iterations:

![spiral image]({{https://blbadger.github.io}}pendulum_map/spiral_map_0.9t.png)


And when $\Delta t = 0.9999$, this effect is so pronounced that there appears to be a ring attractor,

![spiral image]({{https://blbadger.github.io}}pendulum_map/spiral_map_0.9999t.png)


But this is not so!  Closer inspection of this ring reveals that there is no change in point density between the starting and ending ring: instead, meaning that the points are still moving towards the origin at a constant rate.

![spiral image]({{https://blbadger.github.io}}pendulum_map/spiral_map_zoom.png)

Only at $\Delta t = 1$ is there a 1-dimensional attractor, but this is unstable: at small values less than or greater than 1, iterations head towards the origin or else towards infinity. The linear system yeilds a 1-dimensional ring map only when the starting coordinate is already on the ring, and thus it is incapable of forming a 1-dimensional attractor (ie a stable set) as was the case for the nonlinear system.


###  Pendulum maps with 1-dimensional attractors have fractal wave patterns

Take $\Delta t$ to be 0.04087, which produces a similar map to that found above for $\Delta t= 0.04088$.  Now let's zoom in on the upper part of the map:

![pendulum image]({{https://blbadger.github.io}}pendulum_map/pendulum_0.0487t_zoom1.png)
![pendulum image]({{https://blbadger.github.io}}pendulum_map/pendulum_0.0487t_zoom2.png)
![pendulum image]({{https://blbadger.github.io}}pendulum_map/pendulum_0.0487t_zoom3.png)
![pendulum image]({{https://blbadger.github.io}}pendulum_map/pendulum_0.0487t_zoom4.png)

Notice that more and more waves are visible as the scale decreases: the wave pattern is a fractal.  

Waves are not observed for the linear map at any $\Delta t$ size (here at 0.9999):
![pendulum image]({{https://blbadger.github.io}}pendulum_map/swirl_map_zoom.png)

### Reduction of a Clifford system to the pendulum map

There are a number of similarities between widely different nonlinear systems.  Perhaps the most dramatic example of this is the ubiquitous appearance of self-similar fractals in chaotic nonlinear systems (as seen above).  This may be most dramatically seen when the constant parameters of certain equation systems are tweaked such that the output produces a near-copy of another equation system, a phenomenon that is surprisingly common to nonlinear systems. For example, take the Clifford attractor:

$$
x_{n+1} = sin(ay) + c \cdot cos(ax) \\
y_{n+1} = sin(bx) + d \cdot cos(by) 
\tag{3} $$

This is clearly and very different equation system than one modeling pendulum swinging, and for most constant values it produces a variety of maps that look nothing like what is produced by the pendulum system.  But observe what happens when we iterate semicontinuously, setting

$$
a=-0.3, b=0.2, c=0.5, d=0.3, \Delta t = 0.9 \\
(x_0, y_0) = (90, 90)
$$

We have a (slightly oblong) pendulum map!

![clifford pendulum image]({{https://blbadger.github.io}}pendulum_map/clifford_pendulum.png)

---

### If using large $\Delta t$ values yeilds a physically inaccurate map, what do these images mean?

There are some physically relevant reasons to increase a $\Delta t$ value: 

1. The case of periodic forcing, where external energy is applied to a physical system in regular intervals.  The *dt* value may be thought of as a direct measure of this energy, as a large enough *dt* will send this system towards infinity (ie infinite velocity). 

2. When a field is intermittent: if a particle moves smoothly but only interacts with a field at regular time intervals, the same effect is produced.


The real utility in increasing $\Delta t$ is to reveal the intricacies of nonlinear systems in two dimensions.  








