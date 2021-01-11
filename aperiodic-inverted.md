## Periodicity and reversibility

Suppose one were to observe movement over time, and wanted to describe the movement mathematically in order to be able to predict what happens in the future.  This is the goal of dynamical theory, and other pages on this site should convince one that precise knowledge of the future is much more difficult than it would seem even when a precise dynamical equation is known.  

What about if someone were curious about the past?  Given a system of dynamical equations describing how an object's movement occurs, can we find out where the object came from?  This question is addressed here for two relatively simple maps, the logistic and Henon.  

### The logistic map is non-invertible

The logistic equation, which has been explored [here](https://blbadger.github.io/logistic-map.html) and [here](https://blbadger.github.io/logistic-boundary.html) is as follows:

$$
x_{n+1} = rx_n(1-x_n)
\tag{1}
$$

The logistic equation is a one-dimensional discrete dynamical map of very interesting behavior: periodicity for some values of $r$, aperiodicity for others, and for $0 < r < 4$, the interval $(0, 1)$ is mapped into itself. 

What does (1) look like in reverse, or in other words given any value $x_{n+1}$ which are the values of $x_n$ which when evaluated with (1) yield $x_{n+1}$?  Upon substituting $x_n$ for $x_{n+1}$, 

$$
x_n = rx_{n+1}(1-x_{n+1}) \\
0 = -rx_{n+1}^2 + rx_{n+1} - x_n \\
$$

The term $x_n$ can be treated as a constant (as it is constant for any given input value $x_n$), and therefore we can solve this expression for $x_{n+1}$ with the quadratic formula $ax^2 + bx + c$ where $a = -r$, $b = r$, and $c = -x_n$, to give

$$
x_{n+1} = \frac{r \pm \sqrt{r^2-4rx_n}}{2r}
\tag{2}
$$

Now the first thing to note is that this dynamical equation is not strictly a function: it maps a single input $x_n$ to two outputs $x_{n+1}$ (one value for $+$ or $-$ taken in the numerator) for many values of $x_n \in (0, 1)$ whereas a function by definition has one output for any input that exists in the pre-image set.  In other words the logistic map is non-invertible.

### The aperiodic logistic map in reverse is unstable

Suppose one wanted to calculate the all the possible values that could have preceded $x_n$ after a certain number of steps.  The set `s` of points a point could have come from after a given number of `steps` may be found using recursion on (2) as shown below:

```python
def reverse_logistic_map(r, array, steps, s):
	# returns a set s of all possible values of
	# the reverse logistic starting from array[0]
	# after a given number of steps
	if steps == 0:
		for i in array:
			s.add(i)
		return 

	array_2 = []
	for y in array:
		y_next = (r + (r**2 - 4*r*y)**0.5)/ (2*r)
		if not np.iscomplex(y_next):
			if 0 < y_next < 1:
				array_2.append(y_next)

		y_next = (r - (r**2 - 4*r*y)**0.5)/ (2*r)
		if not np.iscomplex(y_next):
			if 0 < y_next < 1:
				array_2.append(y_next)
	
	reverse_logistic_map(r, array_2, steps-1, s)

```

Aside:  At first glance it may seem that having many (a countably infinite number, to be precise) values that eventually meet to make the same trajectory suggests that the logistic map is not sensitive to initial conditions.  This is not so because it requires an infinite number of iterations for these values to reach the same trajectory.  Remembering that aperiodic is another way of saying 'periodic with period $\infty$', this is to be expected.

###  Aperiodic logistic maps are not practically reversible

The logistic map is not because it is 2-to-1 for many values of $x_n \in (0, 1)$: there is no way to know which of the two points $x_{n-1}, x_{n-1}'$ the trajectory actually visited.  For values of r that yield an aperiodic logistic map trajectory, only one of the two $x_{n-1}, x_{n-1}'$ points may be visited because aperiodic trajectories never revisit previous points, and if either point was visited then $x_n$ is also visited.  Therefore aperiodic logistic map trajectories follow only one of the many possible previous trajectories, and which one is impossible to determine.

But what about reversibility with respect to future values $(x_{n+1}, x_{n+2} ...)$?  In other words, given that $x_{n-1}, x_{n-1}'$ both map to the same trajectory $(x_{n+1}, x_{n+2} ...)$, can we find any previous values of $x_n$ that yield the same fucutre trajectory?  This criteria for reversibility is similar to the criteria for one-way functions, which stipulates that either $x_{n-1}, x_{n-1}'$ may be chosen as a suitable previous value and that there is no difference between the two with respect to where they end up.  Is the logistic map reversible under this definition, or equivalently is the logistic map a one way function?

Symbolically, it is not: (2) may be applied any number of times to find the set of all possible values for any previous iteration.  But if this is so, why was it so difficult to use this procedure to find accurate sets of previous values, as attempted in the last section? 

Consider what happens when one attempts to compute successive iterations of (2): first one square root is taken, then another and another etc.  Now for nearly every initial value $x_n$, these square roots are not of perfect squares and therefore yield irrational numbers.  Irrational numbers cannot be represented with complete accuracy in any computation procedure with finite memory, which certainly applies to any computation procedure known.  The definition for 'practical' computation is precisely this: computations that can be accomplished with finite memory in a finite number of steps.  The former stipulation is not found in the classic notion of a Turing machine, but is most accurate to what a non-analogue computer is capable of.

This would not matter much if approximations stayed accurate after many iterations of (2), but one can show this to be untrue for values of r and $x_0$ such that (1) is aperiodic...

### The Henon map is invertible 

The [Henon map](https://blbadger.github.io/henon-map.html) is a two dimensional discrete dynamical system defined as

$$
x_{n+1} = 1 - ax_n^2 + y_n \\
y_{n+1} = bx_n \\
\tag{3}
$$

Let's try the same method used above to reverse $x_{n+1}$ of (2) with respect to time, 

$$
y_{n} = bx_{n+1} \\
x_{n+1} = \frac{y_n}{b}
$$

using this, we can invert $y_{n+1}$ as follows:

$$
y_{n+1} = ax_{n+1}^2 + x_n - 1 \\
y_{n+1} = a \left(\frac{y_n}{b}\right)^2 + x_n - 1
$$

Therefore the inverse of the Henon map is:

$$
x_{n+1} = \frac{y_n}{b} \\
y_{n+1} = a \left( \frac{y_n}{b} \right)^2 + x_n - 1
\tag{4}
$$

This is a one-to-one map, meaning that (3) is invertible.  

Does this mean that, given some point $(x, y)$ in the plane we can determine the path it took to get there?  Let's try this with the inverse function above for $a = 1.4, b=0.3$

```python
#! python3
def reversed_henon_map(x, y, a, b):
	x_next = y / b
	y_next = a * (y / b)**2 + x - 1
	return x_next, y_next
	
array = [[0, 0]]
a, b = 1.4, 0.3
for i in range(4):
	array.append(reversed_henon_map(array[-1][0], array[-1][1], a, b))

print (array)
```
Starting at the origin, future iterations diverge extremely quickly:

```python
[[0, 0], (0.0, -1.0), (-3.3333333333333335, 14.555555555555557), (48.518518518518526, 3291.331961591222), (10971.106538637407, 168511297.67350394)]
```

### The reverse Henon map is unstable

We know that a starting value on the Henon attractor itself should stay bounded if (1) iterating in reverse, at least for the number of iterations it has been located near the attractor.  For example, the following yields a point that has existed near the attractor for 1000 iterations

```python
def henon_map(x, y, a, b):
	x_next = 1 - a*x**2 + y
	y_next = b*x
	return x_next, y_next

x, y = 0, 0
a, b = 1.4, 0.3

for i in range(1000):
	x_next, y_next = henon_map(x, y, a, b)
	x, y = x_next, y_next

# x, y is the coordinate of a point existing near H for 1000 iterations
```

One might not expect this point to diverge until after the 1000 iterations have been reversed, but this is not the case: plugging `x, y` into the initial point for the reverse equation as follows

```python
array = [[x, y]]
for i in range(30):
	array.append(reversed_henon_map(array[-1][0], array[-1][1], a, b))

print (array)
```
results in
```python
[[0.5688254014690528, -0.08255572592986403], (-0.2751857530995468, -0.3251565203383966), (-1.0838550677946555, 0.36945277807827326), (1.231509260260911, 0.03940601355707107), (0.13135337852357026, 0.25566445433028995), (0.8522148477676332, 0.14813158398142479), (0.4937719466047493, 0.19354987712301397), (0.6451662570767133, 0.07650724558327537), (0.25502415194425126, -0.2637814976184484), (-0.8792716587281613, 0.3373902617238522), (1.1246342057461742, -0.10854872330010212), (-0.3618290776670071, 0.3079225997696742), (1.026408665898914, 0.11309157153833649), (0.37697190512778833, 0.225359610056858), (0.7511987001895267, 0.16699118716079653), (0.5566372905359884, 0.1849818026908716), (0.6166060089695721, 0.08892144895232601), (0.29640482984108674, -0.260395838616055), (-0.8679861287201833, 0.3511647173519976), (1.1705490578399922, 0.05027300681394742), (0.16757668937982473, 0.20986378339289535), (0.6995459446429846, -0.14731297048715142), (-0.4910432349571714, 0.03711878667906987), (0.12372928893023291, -1.469610723242318), (-4.898702410807727, 32.71992872244504), (109.06642907481681, 16647.781629174056), (55492.60543058019, 4311201068.530109), (14370670228.433699, 2.8912262794014697e+20), (9.637420931338232e+20, 1.3003183509091478e+42), (4.334394503030493e+42, 2.6301765991061335e+85), (8.767255330353778e+85, 1.0761067243866343e+172)]
```

which demonstrates divergence in around 22 iterations, far fewer than the 1000 we had expected.  

Is the divergence specific to the point we chose above, or do all points near the Henon attractor eventually diverge in a similar manner?  This can be investigated as follows:

```python
def reverse_henon_stability(max_iterations, a, b, x_range, y_range):
	xl, xr = -2.8, 2.8
	yl, yr = 0.8, -0.8

	x_list = np.arange(xl, xr, (xr - xl)/x_range)
	y_list = np.arange(yl, yr, -(yl - yr)/y_range)
	array = np.meshgrid(x_list[:x_range], y_list[:y_range])

	x2 = np.zeros(x_range)
	y2 = np.zeros(y_range)
	iterations_until_divergence = np.meshgrid(x2, y2)

	for i in iterations_until_divergence:
		for j in i:
			j += max_iterations

	not_already_diverged = np.ones(np.shape(iterations_until_divergence))
	not_already_diverged = not_already_diverged[0] < 1000

	for k in range(max_iterations):
		x_array_copied = copy.deepcopy(array[0]) # copy array to prevent premature modification of x array

		# henon map applied to array 
		array[0] = array[1] / b
		array[1] = a * (array[1] / b)**2 + x_array_copied - 1

		r = (array[0]**2 + array[1]**2)**0.5
		diverging = (r > 10000) & not_already_diverged # arbitrarily large number here
		not_already_diverged = np.invert(diverging) & not_already_diverged
		iterations_until_divergence[0][diverging] = k

	return iterations_until_divergence[0]
```

which results (lighter color indicates more iterations occur before divergence)

![divergence]({{https://blbadger.github.io}}misc_images/henon_reversed_scale.png)

The baker's dough topology found by Smale is evident in this image, meaning that each iteration of the forward Henon map can be decomposed into a series of three stretching or folding events as shown [here](https://en.wikipedia.org/wiki/H%C3%A9non_map).  This topology is common for attractors that map a 2D surface to an attractor of 1 < D < 2: the Henon map for a=1.4, b=0.3 is around 1.25 dimensional. 

The attractor for (3) can be mapped on top of the divergence map for (4) as follows:

```python
steps = 100000
X = [0 for i in range(steps)]
Y = [0 for i in range(steps)]

X[0], Y[0] = 0, 0 # initial point

for i in range(steps-1):
	if abs(X[i] + Y[i])**2 < 1000:
		X[i+1] = henon_map(X[i], Y[i], a, b)[0]
		Y[i+1] = henon_map(X[i], Y[i], a, b)[1]

plt.plot(X, Y, ',', color='white', alpha = 0.5, markersize = 0.1)
plt.imshow(reverse_henon_stability(40, a, b, x_range=2000, y_range=1558), extent=[-2.8, 2.8, -0.8, 0.8], aspect= 2.2, cmap='inferno', alpha=1)
plt.axis('off')
plt.savefig('henon_reversed{0:03d}.png'.format(t), dpi=420, bbox_inches='tight', pad_inches=0)
plt.close()
```
which gives

![divergence]({{https://blbadger.github.io}}misc_images/henon_reversed_overlay.png)

where, as expected, the areas of slower divergence align with the attractor (in white). 

From a = 1 to a = 1.5, holding b=0.3 constant,

{% include youtube.html id='gb18hw3ndpU' %}

It is interesting to note that the divergence map for (3) is not simply the inverse of the divergence map for (3), which is presented [here](https://blbadger.github.io/henon-map.html), given that (4) is the inverse of (3).  In particular, regions outside the attractor basin for (3) diverge, meaning that a trajectory starting at say (10, 10) heads to infinity.  But this region also diverges for (4), which is somewhat counter-intuitive given that (4) should give the iterations of (3) in reverse.  

For a=0.2, -1 < b < 0, (3) experiences a point attractor for initial values in the attractor basin: successive iterations spiral in towards the point

$$
x_n = \frac{(b-1) + \sqrt{(b-1)^2 + 4a}}{2a} \\
y_n = b * x_n
$$

outside of which values diverge. For b <= -1, the attractor basin collapses, and nearly all starting points lead to trajectories that spiral out to infinity.  

![divergence]({{https://blbadger.github.io}}misc_images/henon_reversed030.png)

The area in the center does not diverge after 40 iterations.  Do initial points in this area ever diverge?  This question can be addressed by increasing the maximum iterations number.  Doing so from 20 maximum iterations to 520, iterating (4) for a=0.2, b=-0.99 we have

{% include youtube.html id='81ewwor3Lt8' %}

As is the case for a=1.4, b=0.3 so for (3) with a=0.2, b=-0.99, there are unstable points and regions elsewhere diverge.    

### Aperiodic Henon maps are practically irreversible

Is the Henon map reversible?  In the sense that we can define a composition of functions to determine where a previous point in a trajectory was located, the Henon map is reversible as it is 1-to-1 and an inverse function exists.  Reversing the Henon map entails computing (3) for however many reverse iterations are required. 

But earlier our attempts to reverse the Henon map were met with very limited success: values eventually diverged to infinity even if they were located very near the attractor for (3).  Moreover, divergence occurred in fewer iterations that were taken in the original forward trajectory.  This suggests that although the Henon map is invertible, it is not practically invertible.  'Practical" in this definition means computable using finite memory, which is not a stipulation given to Turing machines but is clearly true of any computation one wishes to perform.  

Is the reverse Henon map necessarily impractical?  If it is aperiodic (as is the case for a=1.4, b=0.3) then yes, and this can be proved as follows.  The Henon map, iterated discontinuously, cannot be defined on the rationals (for more information, see [here](https://blbadger.github.io/most-discontinuous.html)).  As reals are uncountably infinite but rationals are countable, all but a negligable portion of values of the Henon attractor $\mathscr H$ are irrational.  Now irrational numbers are of infinite length, and cannot be stored to perfect accuracy in finite memory. 

How do we know that rational approximations of irrational numbers eventually diverge after many iterations of (4)?  This is because of sensitivity to initial conditions, which implies and is implied by aperiodicity (see [here](https://blbadger.github.io/chaotic-sensitivity.html)).  The proof that (4) is sensitive to initial conditions is as follows: (4) is the one-to-one inverse of (3).  Being aperiodic, the trajectory of (3) never revisits a previous point.  Therefore we know that (4) is aperiodic as well, as it never revisits a previous point being that it defines the same trajectory as (3).  As aperiodicity implies arbitrary sensitivity to initial values, (4) is arbitrarily sensitive to initial values. 

Therefore any two starting points $p_1$ and $p_2$, arbitrarily close together will, given enough iterations of (4) separate.  Now consider that every  all but a negligable portion of values on the Henon map itself are practically non-invertible, meaning that the Henon map itself is for practical purposes irreversible.  As points not on $\mathscr H$ are repelled from this attractor, sensitivity to initial conditions results in most points heading away from $\mathscr H$, which is what was observed numerically above.

Another line of reasoning can be used to show why the Henon map is not invertible, assuming finite computation and memory and aperiodic parameter choice.  This is as follows: $\mathscr H$ is not computable, because it requires an infinite number of iterations of $2$ for a starting point not in $\mathscr H$ to end up precisely on $\mathscr H$.  As the precise location of $\mathscr H$ is not known a priori, it cannot be found but only approximated.  Given that any finite approximation of $\mathscr H$ will eventually lead to errors in iterating (4) due to sensitivity to initial conditions, (3) is non-invertible for all practical purposes. 

Where do subsequent iterations of $p_1, p_2$ go once they diverge?  If (3) contains attractor $\mathscr H$, and therefore is a repellor for (4) because these maps are 1-to-1 inverses of each other.  This means that any point near but not $\mathscr H$ will be repelled from $\mathscr H$ given enough iterations of (4).  As nearly every point in $\mathscr H$ is composed of irrational coordinates, a finite representation of any member of this set of points will be near but not on $\mathscr H$ exactly and thus will be repelled over iterations of (4).

The last statement does not necessarily mean that (3) is not practically invertible for periodic trajectories, because any finite number of iterations of (3) could still be reversed with the same number of iterations of (4).  

