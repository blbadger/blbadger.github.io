## Roots of polynomial equations

### There is no closed solution to find roots for most polynomials

Polynomials are equations of the type $ax^b + cx^d + ... = z$ 

At first glance, rooting polynomials seems to be an easy task.  For a degree 1 polynomial $y = ax + b$, setting $y$ to $0$ and solving for x yields $x = -b/a$. For a degree 2 polynomial $y = ax^2 + bx + c$, the closed form expression 

$$
x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}
$$ 

suffices.  There are more references to the constants (all except $c$ are referenced twice) but there is no indication that we cannot make a closed form expression for larger polynomial roots.  For degree 3 and degree 4 polynomials, this is true: closed form root expressions in terms of $a, b, c ...$ may be found even though the expressions become very long. 

It is somewhat surprising then that for a general polynomial of degree 5 or larger, there is no closed equation (with addition, subtraction, multipliction, nth roots, and division) that allows for the finding of a general root.  This is the Abel-Ruffini theorem.

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

with roots at $x=-3, x=1, x=2$.  But now there is more than one point on the real line where Newton's method fails to converge quickly.  We can plot the pattern such points make as follows: darker the point, the faster the convergence. 

![roots]({{https://blbadger.github.io}}/newton-method/newton_real_still.png)

Looking closer at the point $x=-0.8413$, 

![roots]({{https://blbadger.github.io}}/newton-method/newton_real_zoom.gif)

we find a [Cantor set](fractal-geometry.md).  Not only does this polynomial exhibit many values that fail to find a root, as was the case for $x^3 - 1$, but the locations of these values on the real line are not obvious multiples of a number but instead form a fractal pattern.

### Newton's method in the complex plane

The function defined above can be used to apply Newton's method to complex numbers 

```python
print (successive_approximations(2 + 5j, 20))
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[(2+5j), (1.325009908838684+3.3254062623860485j), (0.8644548662679195+2.199047743713538j), (0.5325816440348543+1.4253747693717602j), ... (-0.5+0.8660254037844386j), (-0.5+0.8660254037844387j)]
```

To avoid differentiating polynomials by hand, a 'Calculate' class 

```python
class Calculate:

	def __init__(self, equation, point, differentiate=False):
		self.equation = equation
		self.point = point
		self.diff = differentiate

	def parse(self):
		'''
		Simple iterative parser to prepare a polynomial
		string for evaluation or differentiation.  Only for
		positive-exponent polynomials
		'''
		equation = self.equation
		digits = '0123456789.'
		characters_ls = [i for i in equation]
		characters_ls = ['start'] + characters_ls
		characters_ls.append('end')
		for i in range(len(characters_ls)-1):
			if characters_ls[i] not in digits and characters_ls[i+1] == 'x':
				characters_ls.insert(i+1, '1')
		ls, i = [], 0

		# parse expression into list
		while i in range(len(characters_ls)):
			if characters_ls[i] in digits:
				number = ''
				j = 0
				while characters_ls[i+j] in digits:
					number += (characters_ls[i+j])
					j += 1
				ls.append(float(number))
				i += j
			else:
				ls.append(characters_ls[i])
				i += 1

		return ls

	def differentiate(self):
		'''
		Finds the derivative of a given
		function 'equation' and computes this derivative at
		value 'point'.  Accepts any polynomial with positive
		exponent values.
		'''
		parsed_exp = self.parse()
		ls, point = parsed_exp, self.point

		# differentiate polynomial
		final_ls = []
		for i in range(len(ls)):

			if isinstance(ls[i], float) and ls[i+1] == 'x' or ls[i-1] == '^' and ls[i-2] == 'x':
				final_ls.append(ls[i])

			if ls[i] == 'x':
				if ls[i+1] == '^':
					final_ls[-1] *= ls[i+2]
					if ls[i+2] > 0: # prevent divide by 0 error
						ls[i+2] -= 1 
				final_ls.append(ls[i])

			if ls[i]== 'x' and ls[i+1] != '^':
				final_ls.append('^')
				final_ls.append(0)

			if ls[i] in ['+', '-', '^']:
				final_ls.append(ls[i])

		while True:
			if isinstance(final_ls[-1], float):
				break
			final_ls.pop()
		final_ls.append('+')

		return final_ls


	def evaluate(self):
		'''
		A helper function that finds the derivative of a given
		function 'equation' and computes this derivative at
		value 'point'. Note that this point may also be an ogrid
		value, in which case the derivative is computed at each
		point on the grid. Accepts any polynomial with positive
		exponent values.
		'''
		if self.diff:
			final_ls = self.differentiate()
		else:
			final_ls = self.parse()

		if final_ls[0] != 'start':
			final_ls = ['start'] + final_ls
		if final_ls[-1] != 'end':
			final_ls.append('end')
		final_ls[0], final_ls[-1] = '+', '+' # change 'start' and 'end' to appropriate markers

		point =self.point
		# evaluate parsed expression
		i = 0
		final_blocks = [[]]
		while i in range(len(final_ls)):
			ls = []
			j = 0
			while final_ls[i+j] not in ['+', '-']:
				ls.append(final_ls[i+j])
				j += 1
			if final_ls[i-1] == '-':
				if ls:
					ls[0] = -1 * ls[0]
			final_blocks.append(ls)
			i += j + 1

		total = 0
		for block in final_blocks:
			if block:
				if '^' not in block:
					if 'x' not in block:
						block += ['x', '^', 0]
					else:
						block += ['^', 1]

				start = block[0] * point ** block[-1]
				total += start

		return total
```

Now a map for how long it takes for each point in the complex plane to become rooted using Newton's method may be generated as follows:

```python	
def newton_raphson_map(equation, max_iterations, x_range, y_range, t):
	print (equation)
	y, x = np.ogrid[-5: 5: y_range*1j, -5: 5: x_range*1j]
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

Note that for this color map, purple corresponds to fast convergence and white slow, with no convergence (within the allotted iterations) in black for emphasis.

What happens when the polynomial goes from a linear form to a nonlinear?  Here $z^1-1 \to z^4-1$

![roots]({{https://blbadger.github.io}}/newton-method/newton_3_ranged.gif)

And taking a closer look at the transition $z^2.86 \to z^2.886$

![roots]({{https://blbadger.github.io}}/newton-method/newton_2.86.gif)

### A simple unrootable polynomial

Arguably the simplest polynomial that by Galois theory does not have a closed form rooted expression is

$$
y = x^5 - x - 1
$$


### Beyond polynomials












