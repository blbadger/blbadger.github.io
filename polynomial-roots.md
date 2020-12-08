## Roots of polynomial equations

Polynomials are equations of the type $ax^n + bx^{n-1} + cx^{n-2} ... + z$ 

At first glance, rooting polynomials seems to be an easy task.  For a degree 1 polynomial $y = ax + b$, setting $y$ to $0$ and solving for x yields $x = -b/a$. For a degree 2 polynomial $y = ax^2 + bx + c$, the closed form expression 

$$
x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}
$$ 

suffices.  There are more references to the constants (all except $c$ are referenced twice) but there is no indication that we cannot make a closed form expression for larger polynomial roots.  For degree 3 and degree 4 polynomials, this is true: closed form root expressions in terms of $a, b, c ...$ may be found even though the expressions become very long. 

It is somewhat surprising then that for a general polynomial of degree 5 or larger, there is no closed equation (with addition, subtraction, multiplication, nth roots, and division) that allows for the finding of all roots.  This is the Abel-Ruffini theorem.

### Newton's method for estimating roots of polynomial equations

What should one do to find the roots to a polynomial, if most do not have closed form root equations?  If the goal is simply to find a root, rather than all roots, we can borrow the Newton-Raphson method from analysis. The procedure, described [here](https://en.wikipedia.org/wiki/Newton%27s_method), involves first guessing a point near a root, and then finding the x-intercept of the line tangent to the curve at this point.  These steps are then repeated iterativley such that the x-intercept found previously is the x-value of the new point.  This can be expressed dyamically as follows:

$$
x_{n + 1} = x_n - \frac{f(x_n)}{f'(x_n)}
$$

After a certain number of iterations, this method settles on a root as long as our initial guess is reasonable.  Let's try it out on the equation $y = x^3 - 1$, which has one real root at $x = 1$.  By tracking each iterative root guess, we can see if the method converges on a value.

```python
#! python3

def successive_approximations(x_start, iterations):
	'''
	Computes the successive approximations for the equation
	y = x^3 - 1.
	'''
	x_now = x_start
	ls = []
	for i in range(iterations):
		f_now = x_now**3 - 1
		f_prime_now = 3*x_now**2 

		x_next = x_now - f_now / f_prime_now
		ls.append(x_now)
		x_now = x_next
    
	return ls
```

Let's try with an initial guess at $x=-50$, 

```python
print (successive_approximations(-50, 20))
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[-50, -33.333200000000005, -22.221833330933322, -14.813880530329783, -9.874401411977551, -6.579515604587147, -4.378643733243303, -2.9017098282008416, -1.894884560449859, -1.1704210552983418, -0.5369513549799065, 0.7981682581594858, 1.0553388026381803, 1.002851080311187, 1.0000080978680779, 1.0000000000655749, 1.0, 1.0, 1.0, 1.0]
```

There is convergence on the (real) root, in 17 iterations.  What about if we try an initial guess closer to the root, say at $x=2$?
```python
print (successive_approximations(2, 10))
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[2, 1.4166666666666665, 1.1105344098423684, 1.0106367684045563, 1.0001115573039492, 1.0000000124431812, 1.0000000000000002, 1.0, 1.0, 1.0]
```

The method converges quickly on the root of 1. Now from this one might assume that starting from the point near $x=0$ would result in convergence in around 8 iterations as well, but 

```python
print (successive_approximations(0.000001, 20))
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[1e-06, 333333333333.3333, 222222222222.2222, 148148148148.14813, 98765432098.76541, 65843621399.17694, 43895747599.451294, 29263831732.96753, 19509221155.311684, 13006147436.874454, 8670764957.916304, 5780509971.944202, 3853673314.6294684, 2569115543.0863123, 1712743695.3908749, 1141829130.2605832, 761219420.173722, 507479613.44914806, 338319742.29943204, 225546494.86628804]
```

a root is not found in 20 iterations! It takes 72 to find converge on a root with this starting point.  By observing the behavior of Newton's method on three initial points, it is clear that simply distance away from the root does not predict how fast the method will converge. Note that the area near $x=0$ is not the only one that fails to converge: negative multiples of $0.7937...$ also fail to converge on a root.

Now consider the equation

$$
y = (x-2)(x-1)(x+3) = x^3-7x+6
$$

with roots at $x=-3, x=1, x=2$.  Again there is more than one point on the real line where Newton's method fails to converge quickly.  We can plot the pattern such points make as follows: darker the point, the faster the convergence. Lines of color corresponding to points on the real line are used for clarity.

![roots]({{https://blbadger.github.io}}/newton-method/real_newton_still.png)

This plot seems reasonable, as the points near the roots converge quickly.  Looking closer at the point $x=-0.8413$, 

![roots]({{https://blbadger.github.io}}/newton-method/newton_real_zoom.gif)

we find a [Cantor set](fractal-geometry.md).  Not only does this polynomial exhibit many values that fail to find a root, as was the case for $x^3 - 1$, but the locations of these values on the real line are not obvious multiples of a number but instead form a fractal pattern.

### Newton's method in the complex plane

The function defined above can be used to apply Newton's method to the equation $z^3-1$ for any complex number, for example

```python
print (successive_approximations(2 + 5j, 20))
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[(2+5j), (1.325009908838684+3.3254062623860485j), (0.8644548662679195+2.199047743713538j), (0.5325816440348543+1.4253747693717602j), ... (-0.5+0.8660254037844386j), (-0.5+0.8660254037844387j)]
```

This means that the complex plane may be explored. Because Newton's method requires evaluation and differentiation of a polynomial, I wrote a class `Calculate` to accomplish these tasks, starting from a polynomial written as a string (which may be found [here](https://github.com/blbadger/polynomial_roots/blob/main/Calculate.py)).   Now a map for how long it takes for each point in the complex plane to become rooted using Newton's method may be generated as follows:

```python
# libraries
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('dark_background')
from Calculate import Calculate # see above

def newton_raphson_map(equation, max_iterations, x_range, y_range, t):
	print (equation)
	# top left to bottom right
	y, x = np.ogrid[5: -5: y_range*1j, -5: 5: x_range*1j]
	z_array = x + y*1j

	iterations_until_rooted = max_iterations + np.zeros(z_array.shape)

	 # create a boolean table of all 'true'
	not_already_at_root = iterations_until_rooted < 10000

	for i in range(max_iterations):
		previous_z_array = z_array
		z = z_array
		f_now = Calculate(equation, z, differentiate=False).evaluate()
		f_prime_now = Calculate(equation, z, differentiate=True).evaluate()
		z_array = z_array - f_now / f_prime_now

		# the boolean map is tested for rooted values
		found_root = (abs(z_array - previous_z_array) < 0.0000001) & not_already_at_root
		iterations_until_rooted[found_root] = i
		not_already_at_root = np.invert(found_root) & not_already_at_root

	return iterations_until_rooted

```

The idea here is to start with a grid of the complex plane points that we are testing (`z_array`), a numerical table `iterations_until_rooted` tracking how many iterations each point takes to find a root and a boolean table `not_already_at_root` corresponding to each point on the plane (set to `True` to start with) to keep track of which points have found a root. An small value (not 0, due to round-off error) which is `0.0000001` then compared to the difference between successive iterations of Newton's method in order to see if the point has moved.  If the point is stationary, the method has found a root and the iteration number is recorded in the `iterations_until_rooted` table.  Note that we are only concerned with whether a root has been found, not which root has been found.

Now we can call this function to see how long it takes to find roots for $x^3 - 1$ as follows

```python
plt.imshow(newton_raphson_map('x^3-1', 50, 1558, 1558, 30), extent=[-5, 5, -5, 5], cmap='twilight_shifted')
plt.axis('on')
plt.show()
plt.close()
```
which yields

![roots]({{https://blbadger.github.io}}/newton-method/Newton_Raphson.png)

Note that for this color map, purple corresponds to fast convergence and white slow, with no convergence (within the allotted iterations) in black for emphasis.  John Hubbard was the first person to observe the fractal behavior of Newton's method in the complex plane, and the border between the set of non-diverging and diverging points may be considered a [Julia set](https://blbadger.github.io/julia-sets.html).

What happens when the polynomial goes from a linear form to a nonlinear?  Here $z = z^1-1 \to z = z^4-1$, follow the link to see the change (right click the video while playing to loop)

{% include youtube.html id='YIO_w4x1P2k' %}

And taking a closer look at the transition 

$$
z^{2.86} - 1 \to z^{2.886} - 1
$$

{% include youtube.html id='qS8g6m0QOik' %}

### A simple unrootable polynomial

Arguably the simplest polynomial that by Galois theory does not have a closed form rooted expression is

$$
y = x^5 - x - 1
$$

Let's explore where roots are found with Newton's method in the complex plane, ie for $z^5-z-1$.  With a scale of  $(-1.845953, 2.154047)$ for the real values on the horizontal axis and $(-1.864322i, 2.135678i)$ on the vertical (with our original color map),

![still]({{https://blbadger.github.io}}/newton-method/newton_x5_still.png)

Zooming in on the point $0.154047 + 0.135678i$, 

{% include youtube.html id='ZTMaKLmLxJM' %}

Small changes in exponents lead to dramatic changes for where roots may be found. At $z^5-z^{1.0046}-1$ (right click to view in higher resolution)

![still]({{https://blbadger.github.io}}/newton-method/Newton_vanishing_still_083.png)

and at $z^5-z^{1.0066}-1$

![still]({{https://blbadger.github.io}}/newton-method/Newton_vanishing_still_119.png)

Upon incrementing $z^5-z-1 \to z^5-z^{1.033}-1$,

{% include youtube.html id='Wi0EQ7WqJtU' %}

Using the identity $e^{\pi i} + 1 = 0$, we can rotate the function in the complex plane about the origin in a radius of $1/4$ as follows:

```python
...
for i in range(max_iterations):
		previous_z_array = z_array
		z = z_array
		f_now = z**5 - z - 1 + np.exp(3.1415j * (t/300))/4 
		f_prime_now = 5*z**4 - 1
		z_array = z_array - f_now / f_prime_now
```

![transition]({{https://blbadger.github.io}}/newton-method/newton_rotated.gif)

We can also perform a polynomial rotation as follows:
```python
...
for i in range(max_iterations):
		previous_z_array = z_array
		z = z_array
		f_now = (np.exp(3.1415j * (t/450000))/4) * z**5 - z * np.exp(3.1415j * (t/450000))/4 - 1 + np.exp(3.1415j * (t/450000))/4 
		f_prime_now = 5 * (np.exp(3.1415j * (t/450000))/4)*z**4 - np.exp(3.1415j * (t/450000))/4
		z_array = z_array - f_now / f_prime_now
```
which starts as

![rotation]({{https://blbadger.github.io}}/newton-method/Newton_all_000.png)

the cover photo for this page found [here](https://blbadger.github.io/) is at t=46, and at t=205 (0.00023 of a full rotation) yields

![rotation]({{https://blbadger.github.io}}/newton-method/Newton_all_205.png)

Follow the link for a video of the rotation:

{% include youtube.html id='NgZZq32in7g' %}

### Incrementing a polynomial power

For the equation

$$
z^7-z-1
$$

with roots estimated via Newton's method (centered on the origin), we have

![dimension]({{https://blbadger.github.io}}/newton-method/newton_z^7-z-1.png)

at $z^{7.11}-z-1$, 

![dimension]({{https://blbadger.github.io}}/newton-method/newton_z^7.11-z-1.png)

and at $z^{7.14}-z-1$ 

![dimension]({{https://blbadger.github.io}}/newton-method/newton_z^7.14-z-1.png)

From $z^7-z-1$ to $z^{7.397}-z-1$,

{% include youtube.html id='fpKBzto_ZnA' %}

### Secant method 

There are other methods besides that of Newton for finding roots of an equation.  Some are closely related to Newton's method, for example see the [secant method](https://en.wikipedia.org/wiki/Secant_method) in which is analagous to a non-analytic version of Newton's method, in which two initial guesses then converge on a root. 

There are two initial guesses, $x_0$ and $x_1$, which are then used to guess a third point

$$
x_{n+1} = x_n - f(x_n) \frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})}
$$

And so on until the next guess is the same as the current one, meaning that a root has been found.  This is algebraically equivalent to Newton's method if the derivative is treated as a discrete quantity, ie

$$
f'(x) = \frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})}
$$

Because there are two initial points rather than one, there is more variety in behavior for any one guess.  To simplify things, here the first guess is half the distance to the value $1+0i$ from the second, and the second guess ($x_1$) is the one that is plotted in the complex plane. This can be accomplished as follows:

```python
def secant_method(equation, max_iterations, x_range, y_range, t):
	...
	# create a boolean table of all 'true'
	not_already_at_root = iterations_until_rooted < 10000
	zeros = np.zeros(z_array.shape) 
	z_last = (z_array - zeros)/2 # setting the initial guess to half the distance to the origin from the second guess, which is plotted

	for i in range(max_iterations):
		previous_z_array = z_array
		z = z_array
		f_previous = Calculate(equation, z_last, differentiate=False).evaluate()
		f_now = Calculate(equation, z, differentiate=False).evaluate()
		z_array = z - f_now * (z - z_last)/(f_now - f_previous) # secant method recurrence

		# the boolean map is tested for rooted values
		...
		z_last = z # set z(-2) to z(-1)
	return iterations_until_rooted
```

For $z^3-1$, 

![secant]({{https://blbadger.github.io}}/newton-method/secant_z^3-1_half.png)

Remembering that black denotes locations that do not converge within the maximum interations (here 50), it is clear that there are more points that fail to find a root using this method compared to Newton's, which is expected given that the convergence rate is slower than for Newton's.

Rotated at a radius of $1/8$ (code found [here](https://github.com/blbadger/polynomial_roots/blob/main/secant_rotated.py)), $z^5-z-1$ yields

{% include youtube.html id='6bdghiTen0s' %}

### Halley's method

Edmond Halley (of Halley's comet fame) introduced the following algorithm to find roots for polynomial equations:

$$
x_{n+1} = x_n - \frac{2f(x_n)f'(x_n)}{2(f'(x_n))^2-f(x_n)f''(x_n)}
\;
$$

The general rate of convergence for this method is cubic rather than quadratic as for Newton's method. This can be accomplished using [Calculate](https://github.com/blbadger/polynomial_roots/blob/main/Calculate.py) as follows:

```python
...
	for i in range(max_iterations):
		previous_z_array = z_array
		z = z_array
		f_now = Calculate(equation, z, differentiate=False).evaluate()
		f_prime_now = Calculate(equation, z, differentiate=True).evaluate()
		diff_string = Calculate(equation, z, differentiate=True).to_string()
		f_double_prime_now = Calculate(diff_string, z, differentiate=True).evaluate()
		z_array = z - (2*f_now * f_prime_now / (2*(f_prime_now)**2 - f_now * f_double_prime_now))
		...
```

For $z^3-1$ the pattern in the complex plane of areas of slow convergence is similar but not identical to that observed for Newton's method (see above)

![Halley]({{https://blbadger.github.io}}/newton-method/halley_x^3-1.png)

For larger polynomials, areas of slow convergence are exhibited.  For $z^{13}-z-1$, 

![Halley]({{https://blbadger.github.io}}/newton-method/halley_x^13-x-1.png)

and incrementing $z^{13}-z-1$ to $z^{14}-z-1$, 

{% include youtube.html id='9xl0BWtcc1Y' %}

And similar shapes to that found for Newton's method are also present.  For $z^{9.067}-z-1$, 

![Halley]({{https://blbadger.github.io}}/newton-method/halley_x^9.067-x-1.png)

and incrementing $z^7-z-1$ to $z^8-z-1$,

{% include youtube.html id='-Kbq3EJploo' %}


