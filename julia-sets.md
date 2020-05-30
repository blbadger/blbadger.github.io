## Julia sets

A Julia (named after Gaston Julia) set is the boundary of the sets of unbounded and bounded iterates of the family of functions

$$
f_a(x) = x^2 + a \; \text{given} \; a, x \in \Bbb C
\tag{1}
$$

where $a$ is fixed and $x_0$ varies about the complex plane $x + yi$.  Different values of $a$ lead to different Julia sets, and together this family of functions $f_a(x) \forall a \in \Bbb C$ are the Julia sets.

This means that any number in the complex plane is in a Julia set if it borders another number $u$ such that iterations of (1) are unbounded

$f^k_a(u) \to \infty \; \mathbf {as} \; k \to \infty$ 

as well as a number $b$ where iterations of (1) are bounded

$f^k_a(c) \not\to \infty \; \mathbf {as} \; k \to \infty$

If we restrict ourselves to the real line, such that $a$ and $x$ are elements of $\Bbb R$, iterations of (1) have a number of interesting features.  Some values of $a$ form Cantor sets (fractal dusts), which may be expected as (1) is a nonlinear equation similar in form to the logistic and Henon maps (see [here](https://blbadger.github.io/logistic-map.html)).   

What happens when we allow $a$ and iterates of (1) to range over the complex plane?  Let's find out! To start with, import the indespensable libraries numpy and matplotlib

```python
#! python3
import numpy as np 
import matplotlib.pyplot as plt 
```

and then define a function to find the values of a Julia set for a certain value of $a$, with a docstring specifying the inputs and outputs of the function.  To view a Julia set, we want an array corresponding to the complex plane with values that border diverging and non-diverging values specially denoted.  One way to do this is to count the number of iterations of (1) any given point in the complex plane takes to start heading towards infinity, and if after a sufficiently large number of iterations the point is still bounded then we can assume that the value will not diverge in the future. 

To make an array of complex numbers, the  class in numpy is especially helpful. We can specify the range of the x (real) and y (imaginary) planes using `ogrid`, remembering that imaginary values are specified with `j` in python.  The array corresponding to the complex plane section is stored as `z_array`, and the value of $a$ is specified (the programs presented here are designed with a value of $ \lvert a \rvert < 2 $ in mind, but can be modified for larger $a$s).  This will allow us to keep track of the values of subsequent iterations of (1) for each point $z$ in the complex plane.  

Also essential is initialization of the array corresponding to the number of iterations until divergence, which will eventually form our picture of the Julia set.

```python3
def julia_set(h_range, w_range, max_iterations):
	''' A function to determine the values of the Julia set. Takes
	an array size specified by h_range and w_range, in pixels, along
	with the number of maximum iterations to try.  Returns an array with 
	the number of the last bounded iteration at each array value.
	'''
	y, x = np.ogrid[-1.4: 1.4: h_range*1j, -1.4: 1.4: w_range*1j]
	z_array = x + y*1j
  a = -0.744 + 0.148j
	iterations_till_divergence = max_iterations + np.zeros(z_array.shape)
  
```

To find the number of iterations until divergence of each point in our array of complex numbers, we can simply loop through the array `z_array` such that each point in the array is 

```python
  for h in range(h_range):
     for w in range(w_range):
        for i in range(max_iterations):
          z = z_array[h][w]
```

It can be shown that values where $\lvert a \rvert > 2$ and $\lvert z \rvert > \lvert a \rvert$, future iterations of (1) inevitably head towards positive or negative infinity. 

This makes it simple to find the number of iterations `i` until divergence: all we have to do is to keep iterating (1) until either the resulting value has a magnitude greater than 2 (as $z$ is complex, we can calculate its magnitude by multiplying $z$ by its conjugate $z^* $ and seeing if this number is greater than $2^2 = 4$.  If so, then we know the number of iterations taken until divergence and we assign this number to the 'iterations_till_divergence' array a the correct index. 

```python
          for i in range(max_iterations):
            z = z**2 + a
            if z * np.conj(z) > 4:
              iterations_till_divergence[h][w] = i
              break
              
	return iterations_till_divergence
```

This will give us an iteration number for each point in `z_array`.  What if (1) does not diverge for any given value? The final array `iterations_till_divergence` is initialized with the maximum number of iterations everywhere, such that if this value is not replaced then it remains after the loops, which signifies that the maximum number of iterations performed was not enough to cause the value to diverge.  Outside all these loops, we return the array of iterations taken.

Now we can plot the `iterations_till_divergence` array!  It is a list of lists of numbers between 0 and 'max_iterations', so we can assign it to a color map (see [here](https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html) for an explanation of matplotlib color maps).  Remembering that the Julia set is the border between bounded and unbounded iterates of (1), a cyclical colormap emphasizing intermediate values (ie points that do not diverge quickly but do eventually diverge) is useful. 

```python
plt.imshow(julia_set(500, 500, 70), cmap='twilight_shifted')
plt.axis('off')
plt.show()
plt.close()
```

Now we can plot the image! 

![julia set1]({{https://blbadger.github.io}}fractals/julia_set1.png)

The resolution settings (determined by h_range and w_range) are not very high in the image above because this program is very slow: it sequentially calculates each individual point in the complex array.  The image above took nearly 10 minutes to make!  

Luckily we can speed things up substantially by calculating many points simultaneously.  The idea is to apply (1) to every value of `z_array` at once, and make a boolean array corresponding to the elements of `z_array` that have diverged at each iteration.  The complex number array setup is the same as above:

```python
import numpy as np 
import matplotlib.pyplot as plt 

def julia_set(h_range, w_range, max_iterations):
	''' A function to determine the values of the Julia set. Takes
	an array size specified by h_range and w_range, in pixels, along
	with the number of maximum iterations to try.  Returns an array with 
	the number of the last bounded iteration at each array value.
	'''
	y, x = np.ogrid[-1.4: 1.4: h_range*1j, -1.4: 1.4: w_range*1j]
	z_array = x + y*1j
	a = -0.744 + 0.148j
	iterations_till_divergence = max_iterations + np.zeros(z_array.shape)
```

Instead of examining each element of ```z_array``` individually, we can use a single loop to iterate (1) as follows:
 
 ```python
	for i in range(max_iterations):
		z_array = z_array**2 + a
 ```

In this loop, we can check if any element of ```z_array``` has diverged,
$|z| > 2, z(z^* ) > 2^2 = 4$
and make a boolean array, with a `True` for each position in `z_array` that has diverged.  Then we add the iteration number to our `iterations_till_divergence` array if that position has diverged, as noted in our `divergent_array` boolean array. 

```python
		z_size_array = z_array * np.conj(z_array)
		divergent_array = z_size_array > 4

		iterations_till_divergence[divergent_array] = i
 ```

If a position has a magnitude greater than 2, we can set its to be 0 to prevent that element from going to infinity by assigning each position of the `z_array` that has a `True` value for the `divergent_array` to be 0.  Finally the array of the number of iterations at each position is returned and an image is made, 

```python
		z_array[divergent_array] = 0 


	return iterations_till_divergence

plt.imshow(julia_set(2000, 2000, 70), cmap='twilight_shifted')
plt.axis('off')
plt.show()
plt.close()
```

which yeilds

![julia set1]({{https://blbadger.github.io}}fractals/julia_set2.png)

This is much faster: it takes less than a second for my computer to make the low-resolution image that previously took nearly ten minutes! Using `cmap='twilight'`, we have

![julia set1]({{https://blbadger.github.io}}fractals/Julia_set_inverted.png)

As Gaston Julia found long ago, these sets bounded but are nearly all of infinite length.  Nowadays we call them fractals because they have characteristics of multiple dimensions: like 1-dimensional lines they don't seem to have width, but like 2-dimensional surfaces they have infinite length in a finite area.  Fractals are defined by having a counting dimension (Hausdorff, box, self-similarity etc) greater then their topological dimension, and nearly all fractals have fractional dimension (3/2, 0.616 etc). 

To put it in another way, fractals stay irregular over different size scales.  They can be spectacularly self-similar (where small pieces are geometrically similar to the whole) like many Julia sets and the Mandelbrot set, but most are not (see this excellent video by 3B1B on the definition of a fractal [here](https://www.youtube.com/watch?v=gB9n2gHsHN4).  The fractals formed by the [Clifford attractor](/clifford-attractor.md) and [pendulum maps](/pendulum-map.md) are not self-similar in the strictest sense.

How can a bounded line possibly have infinite length? If we zoom in on a point, we can see why this is: the set stays irregular at arbitrary scale.  Take the Julia set with $a = -0.29609091 + 0.62491i$.  If we zoom in (with more iterations as scale decreases) on the point at $x = 0.041100001 + -0.6583867i$, we have

![julia set1]({{https://blbadger.github.io}}fractals/Julia_set_inverted.png)

The bounded line stays irregular as we zoom in, and if this irregularity continues ad infinitum then any two points on this set are infinitely far from each other.

The Julia set above looks like a coastline, and it turns out that real coastlines are fractals too!  Here is a photo of the Chesapeake bay hanging in the Smithsonian.  Note how it is rough and irregular at a large scale as well as at much smaller scales!  If one tries to estimate the length of the coastline, the result depends very much on the length of the 'ruler' used to determine the path from point a to b. 

![julia set1]({{https://blbadger.github.io}}fractals/chesapeake_bay.png)







