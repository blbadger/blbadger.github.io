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

```
print (successive_approximations(-50, 20))
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[-50, -33.333200000000005, -22.221833330933322, -14.813880530329783, -9.874401411977551, -6.579515604587147, -4.378643733243303, -2.9017098282008416, -1.894884560449859, -1.1704210552983418, -0.5369513549799065, 0.7981682581594858, 1.0553388026381803, 1.002851080311187, 1.0000080978680779, 1.0000000000655749, 1.0, 1.0, 1.0, 1.0]
```

There is convergence on the (real) root, in 17 iterations.  What about if we try an initial guess closer to the root, say at $x=2$?
```
print (successive_approximations(2, 10))
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[2, 1.4166666666666665, 1.1105344098423684, 1.0106367684045563, 1.0001115573039492, 1.0000000124431812, 1.0000000000000002, 1.0, 1.0, 1.0]
```

The method converges quickly on the root of 1. Now from this one might assume that starting from the point near $x=0$ would result in convergence in around 8 iterations as well, but 

```
print (successive_approximations(0.000001, 20))
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[1e-06, 333333333333.3333, 222222222222.2222, 148148148148.14813, 98765432098.76541, 65843621399.17694, 43895747599.451294, 29263831732.96753, 19509221155.311684, 13006147436.874454, 8670764957.916304, 5780509971.944202, 3853673314.6294684, 2569115543.0863123, 1712743695.3908749, 1141829130.2605832, 761219420.173722, 507479613.44914806, 338319742.29943204, 225546494.86628804]
```

a root is not found in 20 iterations! It takes 72 to find converge on a root with this starting point.  By observing the behavior of Newton's method on three initial points, it is clear that simply distance away from the root does not predict how fast the method will converge. 

The only area on the real line that experiences this difficulty for the last poynomial is near $x=0$, and all other values converge. Now consider the equation

$$
y = (x-2)(x-1)(x+3) = x^3-7x+6
$$

with roots at $x=-3, x=1, x=2$.  But now there is more than one point on the real line where Newton's method fails to converge quickly.  We can plot the pattern such points make as follows: darker the point, the faster the convergence. 

![roots]({{https://blbadger.github.io}}/newton-method/Newton000.png)

The dark areas correspond to

Looking closer at the point $x=-0.8413$, 

[insert plot]

we find a [Cantor set](fractal-geometry.md).


### Newton's method in the complex plane

The same function can be used to apply Newton's method to complex numbers 

```
print (successive_approximations(2 + 5j, 20))
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[(2+5j), (1.325009908838684+3.3254062623860485j), (0.8644548662679195+2.199047743713538j), (0.5325816440348543+1.4253747693717602j), ... (-0.5+0.8660254037844386j), (-0.5+0.8660254037844387j)]
```


### A simple unrootable polynomial

Arguably the simplest polynomial that by Galois theory does not have a closed form rooted expression is

$$
y = x^5 - x - 1
$$


### Beyond polynomials












