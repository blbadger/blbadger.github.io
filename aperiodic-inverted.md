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
$$

Now the first thing to note is that this dynamical equation is not strictly a function: it maps a single input $x_n$ to two outputs $x_{n+1}$ (one value for $+$ or $-$ taken in the numerator) for many values of $x_n \in (0, 1)$ whereas a function by definition has one output for any input that exists in the pre-image set.  In other words the logistic map is non-invertible.

### Practical irreversibility in the aperiodic logistic map



### The Henon map is invertible but unstable

The [Henon map](https://blbadger.github.io/henon-map.html) is a two dimensional discrete dynamical system defined as

$$
x_{n+1} = 1 - ax_n^2 + y_n \\
y_{n+1} = bx_n \\
\tag{2}
$$

Let's try the same method used above to reverse the $y_n$ component of (2) with respect to time:

$$
y_{n+1} = ax_{n+1}^2 + x_n - 1 \\
y_{n+1} = a \frac{y_n^2}{b^2} + x_n - 1
$$

and for $x_n$, 

$$
y_{n} = bx_{n+1} \\
x_{n+1} = \pm \frac{y_n}{b}
$$

Therefore the inverse of the Henon map is:

$$
x_{n+1} = \frac{y_n}{b} \\
y_{n+1} = a \frac{y_n^2}{b^2} + x_n - 1
$$

This is a one-to-one map, meaning that (2) is invertible.  

Does this mean that, given some point $(x, y)$ in the plane we can determine the path it took to get there?  Let's try this with the reversal function above

```python
#! python3
def reversed_henon_map(x, y, a, b):
	x_next = y / b
	y_next = a * (y / b)**2 + x - 1
	return x_next, y_next
	
array = [[0, 0]]
for i in range(4):
	array.append(reversed_henon_map(array[-1][0], array[-1][1], a, b))

print (array)
```
Starting at the origin, future iterations diverge extremely quickly:

```python
[[0, 0], (0.0, -1.0), (-3.3333333333333335, 14.555555555555557), (48.518518518518526, 3291.331961591222), (10971.106538637407, 168511297.67350394)]
```

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

where, as expected, the areas of slower divergence align with the attractor (in white)

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
```
![divergence]({{https://blbadger.github.io}}misc_images/henon_reversed_overlay.png)









