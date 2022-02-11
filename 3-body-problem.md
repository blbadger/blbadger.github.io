## The three body problem

### Newtonian mechanics

Newtonian mechanics are often thought to render the motion of planets and stars a solved problem: as the laws of motion have been found, the only thing left to do is to apply them to the heavenly bodies to know where they will go.  Consulting the section on Newtonian mechanics in Feynman's lectures on physics (vol. 1 section 9-9), we find this confident statement:

"Now armed with the tremendous power of Newton's laws, we can not only calculate such simple motions but also, given only a machine to handle the arithmetic, even the tremendously complex motions of the planets, to as high a degree of precision as we wish!"

This statement seems logical at first glance, and was the hope of Laplace and others immediately following Newton (but not, as we shall see, of Feynman).  The most accurate equations of force, momentum, and gravitational acceleration are all fairly simple and for most examples that are taught in school, there are solutions that do not involve time at all.  We can define the differential equations for which there exists a closed (non-infinite, or in other words practical) solution that does not contain any reference to the variable time as 'solved'.  The mechanics most of us learned were problems that were solvable with some calculus, usually by integrating over time to remove that variable.  The solution furthermore must be of finite length, and cannot itself grow as time increases.

If one peruses the curriculum generally taught to people just learning mechanics, a keen eye might spot something curious: the systems considered in the curriculum are all systems of two objects: a planet and a moon, or else the sun and earth.  This is the problem Newton inherited from Kepler, and the solution he found is the one we learn about today. 

But what about 3 bodies, or more?  Newton attempted to find a similar solution to this problem but failed.  This did not deter others, and it seems that some investigators (Laplace in particular) were confident of a solution being just around the corner, even if they could not find one themselves.

The three body problem may be formulated as follows:

$$
a_1 = -Gm_2\frac{p_1 - p_2}{\lvert p_1 - p_2 \rvert ^3} - Gm_3\frac{p_1 - p_3}{\lvert p_1 - p_3 \rvert^3} \\
\; \\
\; \\
a_2 = -Gm_3\frac{p_2 - p_3}{\lvert p_2 - p_3 \rvert ^3} - Gm_1\frac{p_2 - p_1}{\lvert p_2 - p_1 \rvert^3} \\
\; \\
\; \\
a_3 = -Gm_1\frac{p_3 - p_1}{\lvert p_3 - p_1 \rvert ^3} - Gm_2\frac{p_3 - p_2}{\lvert p_3 - p_2 \rvert^3} 
$$
 
where $p_i = (x_i, y_i, z_i)$ and $a_i$ refers to the acceleration of $p_i$, ie 

$$
a_i = (\ddot x_i, \ddot y_i, \ddot z_i)
$$

Note that $\lvert p_1 \rvert$ signifies a vector norm, not absolute value or cardinality. The vector norm is the distance to the origin, calculated in three dimensions as

$$
\lvert p_1 \rvert = \sqrt {x_1^2 + y_1^2 + z_1^2}
$$

The norm of the difference of two vectors may be understood as a distance between those vectors, if our distance function is an arbitrary dimension -extension of the function above.


### Modeling the three body problem

After a long succession of fruitless attempts, Bruns and Poincare showed that the three body problem does not contain a solution approachable with the method of integration used by Newton to solve the two body problem. No one could solve the three body problem, that is, make it into a self-contained algebraic expression, because it is impossible to do so!  

Is any solution possible?  Sundman found that there is an infinite power series that describes the three body problem, and in that sense there is.  But in the sense of actually predicting an orbit, the power series is no help because it cannot directly infer the position of any of the three bodies owing to round-off error propegation from extremely slow convergence ([ref](https://arxiv.org/pdf/1508.02312.pdf)).  From the point of a solution being one in which time is removed from the equation, the power series is more of a problem reformulation than a solution: the time variable is the power series base.  Rather than numerically integrate the differential equations of motion, one can instead add up the power series terms, but the latter option takes an extremely long time to do.  To give an appreciation for exactly how long, it has been estimated that more than $10^{8000000}$ terms are required for calculating the series for one short time step ([ref](http://articles.adsabs.harvard.edu/pdf/1930BuAst...6..417B)). 

For more information, see [Wolfram's notes](https://www.wolframscience.com/reference/notes/972d). 

Unfortunately for us, there is no general solution to the three body problem: we cannot actually tell where three bodies will be at an arbitrary time point in the future, let alone four or five bodies.  This is an inversion with respect to what is stated in the quotation above: armed with the power of Newton's laws, we cannot calculate, with arbitrary precision in finite time, the paths of any system of more than two objects.  

### Bounded trajectories of 3 objects are (almost always) aperiodic

Why is there a solution to the two body problem but not three body problem?  One can imagine that a problem with thousands of objects would be much harder to deal with than two objects, but why does adding only one more object create such a difficult problem?

One way to gain an appreciation for why is to simply plot some trajectories. Let's do this using python. To start with, a docstring is added and the relevant libraries are imported.

```python
#! python3
# A program that produces trajectories of three bodies
# according to Netwon's laws of gravitation

# import third-party libraries
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')
```

Next we can specify the initial conditions for our three bodies.  Here we have bodies with masses of 10, 20, and 30 kilograms, which could perhaps be a few small asteroids orbiting each other.  We then specify their initial positions and velocities, and for simplicity we assume that the bodies are very small such that collisions do not occur.

```python
# masses of planets
m_1 = 10
m_2 = 20
m_3 = 30

# starting coordinates for planets
# p1_start = x_1, y_1, z_1
p1_start = np.array([-10, 10, -11])
v1_start = np.array([-3, 0, 0])

# p2_start = x_2, y_2, z_2
p2_start = np.array([0, 0, 0])
v2_start = np.array([0, 0, 0])

# p3_start = x_3, y_3, z_3
p3_start = np.array([10, 10, 12])
v3_start = np.array([3, 0, 0])
```

Now for a function that calculates the change in velocity (acceleration) for each body, referred to as `planet_1_dv` etc. that uses the three body formulas above.

```python
def accelerations(p1, p2, p3):
	"""
	A function to calculate the derivatives of x, y, and z
	given 3 object and their locations according to Newton's laws
	
	"""

	m_1, m_2, m_3 = self.m1, self.m2, self.m3
	planet_1_dv = -9.8 * m_2 * (p1 - p2)/(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)**3) - \
		       9.8 * m_3 * (p1 - p3)/(np.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2 + (p1[2] - p3[2])**2)**3)

	planet_2_dv = -9.8 * m_3 * (p2 - p3)/(np.sqrt((p2[0] - p3[0])**2 + (p2[1] - p3[1])**2 + (p2[2] - p3[2])**2)**3) - \
		       9.8 * m_1 * (p2 - p1)/(np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)**3)

	planet_3_dv = -9.8 * m_1 * (p3 - p1)/(np.sqrt((p3[0] - p1[0])**2 + (p3[1] - p1[1])**2 + (p3[2] - p1[2])**2)**3) - \
		       9.8 * m_2 * (p3 - p2)/(np.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2 + (p3[2] - p2[2])**2)**3)

	return planet_1_dv, planet_2_dv, planet_3_dv
	
```

A time step size `delta_t` is chosen, which should be small for accuracy, and the number of steps are specified and an array corresponding to the trajectory of each point is initialized.  Varying initial velocites are allowed, so both position and velocity require array initialization.

```python
# parameters
delta_t = 0.001
steps = 200000

# initialize trajectory array
p1 = np.array([[0.,0.,0.] for i in range(steps)])
v1 = np.array([[0.,0.,0.] for i in range(steps)])

p2 = np.array([[0.,0.,0.] for j in range(steps)])
v2 = np.array([[0.,0.,0.] for j in range(steps)])

p3 = np.array([[0.,0.,0.] for k in range(steps)])
v3 = np.array([[0.,0.,0.] for k in range(steps)])

```

The first element of each position and velocity array is assigned to the variables denoting starting coordinates,

```python
# starting point and velocity
p1[0], p2[0], p3[0] = p1_start, p2_start, p3_start

v1[0], v2[0], v3[0] = v1_start, v2_start, v3_start
```

and the velocity and position at each point is calculated for each time step.  For clarity, a two-step approach is chosen here such that the three body eqution is used to calculate the accelerations for each body given its position.  Then the acceleration is applied using Euler's method such that a new velocity `v1[i + 1]` is calculated. Using the current velocity, the new position `p1[i + 1]` is calculated using Euler's formula.

```python
# evolution of the system
for i in range(steps-1):
	# calculate derivatives
	dv1, dv2, dv3 = accelerations(p1[i], p2[i], p3[i])

	v1[i + 1] = v1[i] + dv1 * delta_t
	v2[i + 1] = v2[i] + dv2 * delta_t
	v3[i + 1] = v3[i] + dv3 * delta_t

	p1[i + 1] = p1[i] + v1[i] * delta_t
	p2[i + 1] = p2[i] + v2[i] * delta_t
	p3[i + 1] = p3[i] + v3[i] * delta_t
```

After the loop above runs, we are left with six arrays: three position arrays and three velocity arrays containing information on each body at each time step.  For plotting trajectories, we are only interested in the position at each time.  Some quick list comprehension can separate x, y, and z from each array for the `plt.plot()` method.  p1, p2, and p3 are colored red, white, and blue, respectively.

```python
fig = plt.figure(figsize=(8, 8))
ax = fig.gca(projection='3d')
plt.gca().patch.set_facecolor('black')

plt.plot([i[0] for i in p1], [j[1] for j in p1], [k[2] for k in p1] , '^', color='red', lw = 0.05, markersize = 0.01, alpha=0.5)
plt.plot([i[0] for i in p2], [j[1] for j in p2], [k[2] for k in p2] , '^', color='white', lw = 0.05, markersize = 0.01, alpha=0.5)
plt.plot([i[0] for i in p3], [j[1] for j in p3], [k[2] for k in p3] , '^', color='blue', lw = 0.05, markersize = 0.01, alpha=0.5)

plt.axis('on')

# optional: use if reference axes skeleton is desired,
# ie plt.axis is set to 'on'
ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])

# make panes have the same color as the background
ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0)), ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0)), ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
plt.show()
plt.close()

```
And we are done!  This yields the following map of the trajectories in 3D space. 

![3 body image]({{https://blbadger.github.io}}/3_body_problem/3_body_3_axes.png)

With grid lines removed for clarity (`ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])` removed),

![3 body image]({{https://blbadger.github.io}}/3_body_problem/3_body_3.png)

The orientation of the view can be specified using the interactive matplotlib interface if `plt.show()` is called, and if images are saved directly then `plt.view_init()` should be used as follows:

```python
...
ax.view_init(elev = 10, azim = 40)
plt.savefig('{}'.format(3_body_image), dpi=300)

```

Over time (sped up for brevity) and from a slightly different perspective, the trajectory is

![3 body vid]({{https://blbadger.github.io}}/3_body_problem/three_body_full.gif)


### Poincare and sensitivity to initial conditions

What happens when we shift the starting position of one of the bodies by a miniscule amount? Changing the z position of the third body from $12$ to $12.000001$ yields

![3 body image]({{https://blbadger.github.io}}/3_body_problem/3_body_3_shifted_all.png)

The trajectories are different to what was observed before, but how so?  The new image has different x, y, and z-scales than before and individual trajectories are somewhat difficult to make out.  

If we compare the trajectory of the first body to it's trajectory (here red) in the slightly shifted scenario (blue),

![3 body image]({{https://blbadger.github.io}}/3_body_problem/3_body_shifted_1.png)

from the side and over time, we can see that they are precisely aligned for a time but then diverge, first slowly then quickly.

![3 body vid]({{https://blbadger.github.io}}/3_body_problem/three_body_shifted.gif)

The same is true of the second body (red for original trajectory, blue for the trajectory when the tiny shift to the third body's initial position has been made)

![3 body image]({{https://blbadger.github.io}}/3_body_problem/3_body_shifted_2.png)

And of the third as well.

![3 body image]({{https://blbadger.github.io}}/3_body_problem/3_body_shifted_3.png)

Plotting the distance between each point and its counterpart as follows

```python
distance_2 = []
for i in range(steps):
	distance_2.append(np.sqrt(np.sum([j**2 for j in p2[i] - p2_prime[i]])))

distance_1 = []
for i in range(steps):
	distance_1.append(np.sqrt(np.sum([j**2 for j in p1[i] - p1_prime[i]])))

distance_3 = []
for i in range(steps):
	distance_3.append(np.sqrt(np.sum([j**2 for j in p3[i] - p3_prime[i]])))

fig, ax = plt.subplots()
ax.plot(time, distance_1, alpha=0.5, color='red')
ax.plot(time, distance_2, alpha=0.5, color='grey')
ax.plot(time, distance_3, alpha=0.5, color='blue')
plt.ylim(0, 120)
ax.set(xlabel='time', ylabel='Distance')
plt.show()
plt.close()
```

yields

![3 body image]({{https://blbadger.github.io}}/3_body_problem/three_body_distance.png)


In 1914, Poincare observed that "small differences in initial conditions produce very great ones in the final phenomena" (as quoted [here](https://books.google.com/books?id=vGuYDwAAQBAJ&pg=PA271&lpg=PA271&dq=Poincare+3+body+problem+impossibility+1880&source=bl&ots=yteTecRsK8&sig=ACfU3U2ngm5xUXygi-JdLzpU0bwORuOq7Q&hl=en&sa=X&ved=2ahUKEwiO4JT_86zqAhUlZjUKHYn5Dk8Q6AEwDHoECAwQAQ#v=onepage&q=Poincare%203%20body%20problem%20impossibility%201880&f=false)). 

If we make slightly worse and worse initial measurements, does the inaccuracy of our prediction get worse too?  Surprisingly the answer to this question is often but not always.  To see how, here we iterate the distance measurement above but change the distance between planet 3 and its 'true' value to increase from $\Delta z = 0.00000001 \to \Delta z = 0.00001$ 

![3 body image]({{https://blbadger.github.io}}/3_body_problem/three_body_distance.gif)

Just as for the [logistic map](logistic-map.md), the benefit we achieve for getting better initial accuracies is unpredictable. Sometimes a better initial measurement will lead to larger errors in prediction!

### Divergence and Homoclinic Tangles

We have so far observed the behavior of only one initial set of points in three-dimensional space for each planet.  But it is especially interesting to consider the behavior of many initial points.  One could consider the behavior of all points in $\Bbb R^3$, but this is often difficult to visualize as the trajectories may form something like a solid object.  To avoid this difficulty, we can restrict our initial points to some subspace.

In $\Bbb R^3$ we can choose vectors in two basis vectors, perhaps $x, y$, to vary while the third $z$ stays constant.  Allowing two basis vectors to change freely in $\Bbb R^3$ forms a two-dimensional sheet in that three-dimensional space, or in other words forms a plane. What happens to points embedded in this plane as they observe Newton's laws of gravitation, and in particular do different points in this plane have different sensitivities to initial conditions?

As we have three bodies, we can choose one of these (say `p1`) to allow to start in different locations in a two-dimensional plane while holding the other two bodies fixed in their starting points before observing the trajectory resulting from each of these different starting locations.  As we are interested in divergence, we can once again compare the trajectory of the planet `p1` at each of these starting points to a slightly shifted planet `p1_prime`.  As we are now dealing with an multidimensional array of initial points rather than only three per planet, a slightly different approach is required, found in [this source code](https://github.com/blbadger/threebody/blob/master/three_body_phase_portrait.py) file.

Sensitivity is initialized by changing the 3 by 1 by 1 initial point for planet 1 to an array of size 3 by y_res by x_res for that planet.  For example, if we want to plot 100 points along both the x and y-axis we start with a 3x100x100 size array for `p1`.  

```python
def sensitivity(self, y_res, x_res, steps):
	"""
	Plots the sensitivity to initial values per starting point of planet 1, as
	measured by the time until divergence.
	"""

	delta_t = self.delta_t
	y, x = np.arange(-20, 20, 40/y_res), np.arange(-20, 20, 40/x_res)
	grid = np.meshgrid(x, y)
	grid2 = np.meshgrid(x, y)

	# grid of all -11, identical starting z-values
	z = np.zeros(grid[0].shape) - 11

	# shift the grid by a small amount
	grid2 = grid2[0] + 1e-3, grid2[1] + 1e-3
	# grid of all -11, identical starting z-values
	z_prime = np.zeros(grid[0].shape) - 11 + 1e-3

	# p1_start = x_1, y_1, z_1
	p1 = np.array([grid[0], grid[1], z])
	p1_prime = np.array([grid2[0], grid2[1], z_prime])
		
```
But now we also have to have arrays of identical dimensions for all the other planets, as their trajectories each depend on the initial points of planet 1.  This means that we must make arrays that track the motion of each planet in three-dimensional space for all points in the initial plane of possible planet 1 values.

For example, planet two can be initialized to the same value as in our approach above, or $(0, 0, 0)$ for both position and velocity, with the following:

```python
	p2 = np.array([np.zeros(grid[0].shape), np.zeros(grid[0].shape), np.zeros(grid[0].shape)])
	v2 = np.array([np.zeros(grid[0].shape), np.zeros(grid[0].shape), np.zeros(grid[0].shape)])
		
```
Sensitivity to initial values can be tested by observing if any of the `p1` trajectories have separated significantly from the shifted `p1_prime` values.  This is known as divergence, and the following function creates a boolean array to see which points have separated (diverged):

```python
def not_diverged(self, p1, p1_prime):
	"""
	Find which trajectories have diverged from their shifted values

	Args:
		p1: np.ndarray[np.meshgrid[bool]]
		p1_prime: np.ndarray[np.meshgrid[bool]]

	Return:
		bool_arr: np.ndarray[bool]

	"""
	separation_arr = np.sqrt((p1[0] - p1_prime[0])**2 + (p1[1] - p1_prime[1])**2 + (p1[2] - p1_prime[2])**2)
	bool_arr = separation_arr <= self.distance
	return bool_arr
		
```

Using this function to test whether or not planet 1 has diverged from its shifted planet 1 prime,

```python
def sensitivity(self, y_res, x_res, steps):
	...
	time_array = np.zeros(grid[0].shape)
	# bool array of all True
	still_together = grid[0] < 1e10
	t = time.time()

	# evolution of the system
	for i in range(self.time_steps):
		if i % 100 == 0:
			print (i)
			print (f'Elapsed time: {time.time() - t} seconds')

		not_diverged = self.not_diverged(p1, p1_prime)

		# points still together are not diverging now and have not previously
		still_together &= not_diverged

		# apply boolean mask to ndarray time_array
		time_array[still_together] += 1
```

now we can follow the trajectories of each planet at each initial point.  Note that these trajectories are no longer tracked, as doing so requires an enormous amount of memory.  Instead the points and velocities are simply updated at each time step.

```python
		# calculate derivatives
		dv1, dv2, dv3 = self.accelerations(p1, p2, p3)
		dv1_prime, dv2_prime, dv3_prime = self.accelerations(p1_prime, p2_prime, p3_prime)

		nv1 = v1 + dv1 * delta_t
		nv2 = v2 + dv2 * delta_t
		nv3 = v3 + dv3 * delta_t

		p1 = p1 + v1 * delta_t
		p2 = p2 + v2 * delta_t
		p3 = p3 + v3 * delta_t
		v1, v2, v3 = nv1, nv2, nv3
```

After $50,000$ time steps, initial points on the $x, y$ plane of planet 1 (ranging from $(-20, 20)$ on both x- and y-axes) we have

![homoclinic tangle]({{https://blbadger.github.io}}/3_body_problem/Threebody_divergence_xy.png)

where lighter values indicate earlier divergence. The folded and stretched topology here is known as the horseshoe map, or equivalently a homoclinic tangle. As $t_i = 0 \to t_i =  100,000$ we have

{% include youtube.html id='Vp4r8SfWoEA' %}

It is worth considering what this map tells us.  In a certain region of 2-dimensional space, a planet's starting point may be shifted only slightly to result in a large difference in the earliest time of divergence.  This is equivalent to saying that a planet's starting point, within a certain region of space, may yield an unpredicable (if it is a point of fast divergence) or relatively predictable (if divergence is slower) trajectory, but even knowing which one of these two possibilities will occur is extremely difficult.  

This topology is not special to points on the $x, y$ plane: on the $y, z$ plane ($y$ on the horizontal axis from -20 to 20 and $z$ on the vertical also ranging from -20 to 20) after $50,000$ time steps we have

![homoclinic tangle]({{https://blbadger.github.io}}/3_body_problem/Threebody_divergence_yz.png)

which, going from $t_i = 0 \to t_i =  100,000$ we have

{% include youtube.html id='C5W1OpqIaiI' %}

### A comparison: two body versus three body problems
How does this help us understand the difference between the two body and three body problems?  Let's examine the two body problem as a restricted case of the three body problem: the same differential equations used above can be used to describe the two body problem simply by setting on the of masses to 0.  Lets remove the first object's mass, and also change the second object's initial velocity for a trajectory that is easier to see.

```python
# masses of planets
m_1 = 0
m_2 = 20
m_3 = 30

...

# p2_start = x_2, y_2, z_2
p2_start = np.array([0, 0, 0])
v2_start = np.array([-3, 0, 0])
```
Plotting the trajectories of p2 and p3, we have 

![3 body image]({{https://blbadger.github.io}}/3_body_problem/two_body_1.png)

This plot looks much more regular!  The trajectories form periodic orbits that, like other two body trajectories, lie along a plane.  We can do some fancy rotation in three dimensional space by changing using a second loop after our array-filling loop to show this.

```python

...

for t in range(360):
	fig = plt.figure(figsize=(10, 10))
	ax = fig.gca(projection='3d')
	plt.gca().patch.set_facecolor('black')
	
	plt.plot([i[0] for i in p2], [j[1] for j in p2], [k[2] for k in p2] , '^', color='white', lw = 0.05, markersize = 0.01, alpha=0.5)
	plt.plot([i[0] for i in p3], [j[1] for j in p3], [k[2] for k in p3] , '^', color='blue', lw = 0.05, markersize = 0.01, alpha=0.5)

	plt.axis('on')
	# optional: use if reference axes skeleton is desired,
	# ie plt.axis is set to 'on'
	ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])

	# make panes have the same color as the background
	ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
	ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
	ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
	
	ax.view_init(elev = 20, azim = t)
	plt.savefig('{}'.format(t), dpi=300, bbox_inches='tight')
	plt.close()
	
```

![3 body image]({{https://blbadger.github.io}}/3_body_problem/two_body_rotated_2.gif)

One might raise a question: aren't trajectory crossings not possible for ordinary differential equations?  For the case of a single object moving in space, this is correct, because any trajectory crossing would imply that some point heads toward two different points next, an impossibility.  But as we have two objects, crossings can occur if the other object is in a different place than before.  What is now not possible is for both objects to revisit a previously-occuppied pair of points but then to travel to a new location.

Now let's see what happens when we shift the starting value of one of the points by the same amount as before ($z_3 = 12 \to z_3 = 12.000001$).

![3 body image]({{https://blbadger.github.io}}/3_body_problem/two_body_1_shifted.png)

The trajectories looks the same!  When both original and shifted trajectories of the $p_2$ are plotted, it is clear to see that there is no separation
($z_3 = 12$ in white and $z_3 = 12.000001$ in blue)

![3 body image]({{https://blbadger.github.io}}/3_body_problem/two_body_shifted_2.png)

This means that this trajectory of a two body problem is not sensitive to initial conditions: it is not chaotic.  It turns out that this is true for all two body problems: all are periodic or quasi-periodic, meaning that future trajectories are identical to past trajectories.  This means that we can remove (all but a negligable amount) of the time variable when we integrate these differential equations.  On the other hand, most three body trajectories are aperiodic.  This means that their future trajectories are never exactly like previous ones, meaning that we cannot remove time from the differential equations.  This makes them unsolveable, with respect to a solution that does not consist of adding up time steps from start to finish.

### The three body problem is general 

One can hope that the three (or more) body problem is restricted to celestial mechanics, and that it does not find its way into other fields of study.  Great effort has been expended to learn about the orbitals an electron will make around the nucleus of a proton, so hopefully this knowledge is transferrable to an atom with more than one proton. Unfortunately it does not: any three-dimensional system with three or more objects that operates according to nonlinear forces (gravity, electromagnetism etc.) reaches the same difficulties outlined above for planets. 

This was appreciated by Feynman, who states in his lectures (2-9):

"[Classical mechanics] is deterministic.  Suppose, however, that we have a finite accuracy and do not know exactly where just one atom is, say to one part in a billion.  Then as it goes along it hits another atom...if we start with only a tiny error it rapidly magnifies to a very great uncertainty.... Speaking more precisely, given an arbitrary accuracy, no matter how precise, one can find a time long enough that we cannot make predictions valid for that long a time"

The idea that error magnifies over time is only true of nonlinear systems, and in particular aperiodic nonlinear systems. On this page, we have seen that in a nonlinear system of two planets, error does not magnify whereas it does for the cases observed with three planets.  Therefore measurement error does not necessarily become magnified, but only for aperiodic dynamical systems. This was the salient recognition of Lorenz, who found that aperiodicity and sensitivity to initial conditions (which is equivalent to magnification of error) implied one another.

### Does it matter that we cannot solve the three body problem, given that we can just simulate the problem on a computer?

When one hears about solutions to the three body problem, they are either restricted to a (miniscule) subset of initial conditions or else are references to the process of numerical integration by a computer.  The latter idea gives rise to the sometimes-held opinion that the three body problem is in fact solveable now that high speed computers are present, because one can simply use extremely precise numeric methods to provide a solution.  

To gain an appreciation for why computers cannot solve our problem, let's first pretend that perfect observations were able to be made.  Would we then be able to use a program to calculate the future trajectory of a planetary system exactly?  We have seen that we cannot when small imperfections exist in observation, but what about if these imperfections do not exist?  Even then we cannot, because it appears that Newton's gravitational constant, like practically all other constants, is an irrational number.  This means that even a perfect measurement of G would not help because it would take infinite time to enter into a computer exactly.

That said, computational methods are very good for determining short-term trajectories.  Furthermore, when certain bodies are much larger in mass than others (as is the case in the solar system where the sun is much more massive than all the planets combined), the ability to determine trajectories is substantially enhanced. But like any aperiodic equations system, the ability to determine trajectories for all time is not possible.

