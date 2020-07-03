## The three body problem

### Newtonian mechanics

When I was learning Newtonian mechanics in school, I thought that the subject is an open and shut matter: as the laws of motion have been found, the only thing left to do is to apply them to the heavenly bodies to know where they will go.  I was told that this is the case: indeed, consulting the section on Newtonian mechanics in Feynman's lectures on physics (vol. 1 section 9-9), we find this confident statement:

"Now armed with the tremendous power of Newton's laws, we can not only calculate such simple motions but also, given only a machine to handle the arithmetic, even the tremendously complex motions of the planets, to as high a degree of precision as we wish!"

This statement seems logical at first glance.  The most accurate equations of force, momentum, gravitational acceleration etc. are all fairly simple and for most examples that were taught to me, there were solutions that does not involve time at all.  We can define the differential equations for which there exists a closed (non-infinite, or in other words practical) solution that does not contain any reference to time as 'solved'.  The mechanics I learned were problems that were solveable with some calculus, usually by integrating over time to remove that variable.  

If one peruses the curriculum generally taught to people just learning mechanics, a keen eye might spot something curious: the systems considered in the curriculum are all systems of two objects: a planet and a moon, the sun and earth.  This is the problem Newton inherited from Kepler, and the solution he found is the one we learn about today. But what about 3 bodies, or more?  Newton attempted to find a similar solution to this problem but failed.  This did not deter others, and it seems that some investigators such as Laplace were confident of a solution being just around the corner, even if they could not find one themselves.

And why shouldn't they be?  The three body problem may be formulated as follows:

$$
a_1 = -Gm_2\frac{p_1 - p_2}{\lvert p_1 - p_2 \rvert ^3} - Gm_3\frac{p_1 - p_3}{\lvert p_1 - p_3 \rvert^3} \\
 \\
a_2 = -Gm_3\frac{p_3 - p_2}{\lvert p_3 - p_2 \rvert ^3} - Gm_1\frac{p_2 - p_1}{\lvert p_2 - p_1 \rvert^3} \\
 \\
a_3 = -Gm_3\frac{p_3 - p_1}{\lvert p_3 - p_1 \rvert ^3} - Gm_1\frac{p_3 - p_2}{\lvert p_3 - p_2 \rvert^3} 
$$
 
where $p_i = (x_i, y_i, z_i)$ and $a_i$ refers to the acceleration of $p_i$, ie $a_i = (\ddot x_i, \ddot y_i, \ddot z_i)$.  Note that $\lvert p_1 \rvert$ signifies a vector norm, not absolute value or cardinality. The vector norm is the distance to the origin, calculated in three dimensions as

$$
\lvert p_1 \rvert = \sqrt {x_1^2 + y_1^2 + z_1^2}
$$

The norm of the difference of two vectors may be understood as a distance between those vectors, if our distance function is an arbitrary dimension - extension of the function above.

This function does not seem too unwieldy.  

### Modelling the three body problem

After a long succession of fruitless attempts, Bruns and Poincare showed that the three body problem does not contain a solution approachable with the method of integration used by Newton to solve the two body problem. No one could solve the three body problem, that is make it into a self-contained algebraic expression, because it is impossible to do so!  

Is any solution possible?  Sundman found that there is an infinite power series that describes the three body problem, and in that sense there is.  But in the sense of actually predicting an orbit, the power series is no help because it cannot directly infer the position of any of the three bodies owing to round-off error propegation from extremely slow convergence [ref](https://arxiv.org/pdf/1508.02312.pdf).  From the point of a solution being one in which time is removed from the equation, the power series is more of a problem reformulation than a solution: the time variable is the power series base.  Rather than numerically integrate the differential equations of motion, one can instead add up the power series terms, but the latter option takes an extremely long time to do.  To give an appreciation for exactly how long, it has been estimated that more than $10^{8000000}$ terms are required for calculating the series for one short time step [ref](http://articles.adsabs.harvard.edu/pdf/1930BuAst...6..417B). 

For more information, see Wolfram's notes [here](https://www.wolframscience.com/reference/notes/972d). 

Unfortunately for us, there is no general solution to the three body problem: we cannot actually tell where three bodies will be at an arbitrary time point in the future, let alone four or five bodies.  This is an inversion with respect to what is stated in the quotation above: armed with the power of Newton's laws, we cannot calculate, with arbitrary precision in finite time, the paths of any system of more than two objects.  

### Trajectories of 3 objects are chaotic

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
	'''A function to calculate the derivatives of x, y, and z
	given 3 object and their locations according to Newton's laws
	'''
	planet_1_dv = -9.8 * m_2 * (p1 - p2)/(np.sqrt(np.sum([i**2 for i in p1 - p2]))**3) - 9.8 * m_3 * (p1 - p3)/(np.sqrt(np.sum([i**2 for i in p1 - p3]))**3)

	planet_2_dv = -9.8 * m_3 * (p2 - p3)/(np.sqrt(np.sum([i**2 for i in p2 - p3]))**3) - 9.8 * m_1 * (p2 - p1)/(np.sqrt(np.sum([i**2 for i in p2 - p1]))**3)

	planet_3_dv = -9.8 * m_1 * (p3 - p1)/(np.sqrt(np.sum([i**2 for i in p3 - p1]))**3) - 9.8 * m_2 * (p3 - p2)/(np.sqrt(np.sum([i**2 for i in p3 - p2]))**3)

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
# starting point
p1[0], p2[0], p3[0] = p1_start, p2_start, p3_start

v1[0], v2[0], v3[0] = v1_start, v2_start, v3_start
```
and the velocity and position at each point is calculated for each time step.  For clarity, a two-step approach is chosen here such that the three body eqution is used to calculate the accelerations for each body given its position.  Then the acceleration is applied using Euler's method such that a new velocity `v1[i + 1]` is calculated. Using the current velocity, the new position `p1[i + 1]` is calculated using Euler's formula.

```python
# evolution of the system
for i in range(steps-1):
	#calculate derivatives
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

# make pane's have the same colors as background
ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0)), ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0)), ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
plt.show()
plt.close()
```
And we are done!  This yeilds the following map of the trajectories in 3D space. 

![3 body image]({{https://blbadger.github.io}}/3_body_problem/3_body_3_axes.png)

With grid lines removed for clarity (`ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])` removed),

![3 body image]({{https://blbadger.github.io}}/3_body_problem/3_body_3.png)

Over time and from a slightly different perspective, the trajectory is

[insert gif]


### Poincare and sensitivity to initial conditions

What happens when we shift the starting position of one of the bodies by a miniscule amount? Changing the z position of the third body from $12$ to $12.000001$ yeilds

![3 body image]({{https://blbadger.github.io}}/3_body_problem/3_body_3_shifted_all.png)

The trajectories are different to what was observed before, but how so?  The new image has different x, y, and z-scales than before and individual trajectories are somewhat difficult to make out.  

If we compare the trajectory of the first body to it's trajectory (here red) in the slightly shifted scenario (blue),
![3 body image]({{https://blbadger.github.io}}/3_body_problem/3_body_shifted_1.png)

from the side and over time, we can see that they are precisely aligned for a time but then diverge, first slowly then quickly.

[insert gif]

The same is true of the second body (red for original trajectory, blue for the trajectory when the tiny shift to the third body's initial position has been made)
![3 body image]({{https://blbadger.github.io}}/3_body_problem/3_body_shifted_2.png)

And of the third as well.
![3 body image]({{https://blbadger.github.io}}/3_body_problem/3_body_3_shifted_all.png)


In 1914, Poincare observed that "small differences in initial conditions produce very great ones in the final phenomena" (as quoted [here](https://books.google.com/books?id=vGuYDwAAQBAJ&pg=PA271&lpg=PA271&dq=Poincare+3+body+problem+impossibility+1880&source=bl&ots=yteTecRsK8&sig=ACfU3U2ngm5xUXygi-JdLzpU0bwORuOq7Q&hl=en&sa=X&ved=2ahUKEwiO4JT_86zqAhUlZjUKHYn5Dk8Q6AEwDHoECAwQAQ#v=onepage&q=Poincare%203%20body%20problem%20impossibility%201880&f=false). 

### Phase space portraits of three body trajectories are fractals

"One must be struck with the complexity of this shape, which I do not even attempt to illustrate" (Mandelbrot 1982).

### The three body problem is general 

One can hope that the three (or more) body problem is restricted to celestial mechanics, and that it does not find its way into other fields of study.  Great effort has been expended to learn about the orbitals an electron will make around the nucleus of a proton, so hopefully this knowledge is transferrable to an atom with more than one proton. This hope is in vain: any three-dimensional system with three or more objects that operates according to nonlinear equations reaches the same difficulties outlined above for planets. 


### Does it matter that we cannot solve the three body problem, given that we can just 'solve' the problem on a computer?

When one hears about solutions to the three body problem, they are either restricted to a (miniscule) subset of initial conditions or else are references to the process of numerical integration by a computer.  The latter idea gives rise to the sometimes-held opinion that the three body problem is in fact solveable now that high speed computers are present, because one can simply use extremely precise numeric methods to provide a solution.  

To gain an appreciation for why computers cannot solve our problem, let's first pretend that perfect observations were able to be made.  Would we then be able to use a program to calculate the future trajectory of a planetary system exactly?  We have seen that we cannot when small imperfections exist in observation, but what about if these imperfections do not exist?  Even then we cannot, because it appears that Newton's gravitational constant, like practically all other constants, is an irrational number.  This means that even a perfect measurement of G would not help because it would take infinite time to enter into a computer exactly.



