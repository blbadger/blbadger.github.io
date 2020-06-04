## Mandelbrot set with variations

The Mandelbrot set $\mathscr M$ is the set of points $a$ in the complex plane for which the Julia sets are connected.  This happens to be the same set of points that iterations of the equation

$$
z = z^2 + a
\tag{1}
$$

do not diverge (go to positive or negative infinity) but instead are bounded upon many iterations at a starting value of $z = 0$.  Thus the Mandelbrot set is very similar to the Julia set but instead of fixing $a$ and ranging about $z$, the starting value of $z$ is fixed at 0 and the value of $a$ is ranged about the complex plane. 

Let's compute $\mathscr M$. As for the [Julia sets](/julia-sets.html), the simplest way to do this is to initialize a complex plane as an `ogrid` array in numpy, with the difference being that `a` is assigned to this array, not `z`, which is instead asigned to an identically sized array of 0s.

```python
#! python3
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('dark_background')

def mandelbrot_set(h_range, w_range, max_iterations):
	y, x = np.ogrid[-1.4: 1.4: h_range*1j, -1.8: 1: w_range*1j]
	a_array = x + y*1j
	z_array = np.zeros(a_array.shape)
	iterations_till_divergence = max_iterations + np.zeros(a_array.shape)
```

Now we can iterate at each position of `z_array` until the value diverges, using the value of `a` in `a_array`.  Finally, call the function and plot the results!

```python
	for h in range(h_range):
		for w in range(w_range):
			for i in range(max_iterations):
				z = z_array[h][w]
				a = a_array[h][w]
				for i in range(max_iterations):
					z = z**2 + a
					if z * np.conj(z) > 4:
						iterations_till_divergence[h][w] = i
						break

	  return iterations_till_divergence

plt.imshow(mandelbrot_set(800, 800, 30), cmap='twilight_shifted')
plt.axis('off')
plt.show()
plt.close()
```

![mandelbrot image]({{https://blbadger.github.io}}fractals/mandelbrot_custom_800x800x30.png)
  
This method is perfectly good, but slow.  Luckily we can use the numpy `ogrid` to compute divergence much faster! (see the [Julia sets page](/julia-sets.html) for more information)

```python
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('dark_background')


def mandelbrot_set(h_range, w_range, max_iterations):
	y, x = np.ogrid[-1.4: 1.4: h_range*1j, -1.8: 1: w_range*1j]
	a_array = x + y*1j
	z_array = np.zeros(a_array.shape)
	iterations_till_divergence = max_iterations + np.zeros(a_array.shape)

	for i in range(max_iterations):
		# mandelbrot equation
		z_array = z_array**2 + a_array

		# make a boolean array for diverging indicies of z_array
		z_size_array = z_array * np.conj(z_array)
		divergent_array = z_size_array > 4

		iterations_till_divergence[divergent_array] = i

		# prevent overflow (numbers -> infinity) for diverging locations
		z_array[divergent_array] = 0 
    
	return iterations_till_divergence

plt.imshow(mandelbrot_set(2000, 2000, 70), cmap='twilight_shifted')
plt.axis('off')
plt.savefig('mandelbrot.png', dpi=300)
plt.close()
```

This code is perfectly valid for mapping $\mathscr M$, itself (the dark region), but the colors look strange: there is a banding pattern that is not seen in the plot from the other program for $\mathscr M$.  

![ mandelbrot]({{https://blbadger.github.io}}fractals/mandelbrot_diverging2.png)

A little retrospection can convince us that there is a problem with how we compute the first iteration of divergence, mainly that sometimes this program does not actually store the first iteration but instead a later one! Remember that $a$ is being added to $z$ for every iteration even if $z = 0$, as we have set it to stop the values from getting out of hand.  This could cause a later iteration to become larger than 2 (see [here](/julia-sets.md), and the code as it stands would record this later value as the iteration of divergence. This can be remedied by simply changing all values of `a_array` corresponding to just-diverged `z_array` values to be 0

```python
    ...
		# prevent overflow (numbers -> infinity) for diverging locations
		z_array[divergent_array] = 0 

		# prevent the a point from diverging again in future iterations
		a_array[divergent_array] = 0 
```

The colors are accurate now! The above code yeilds

![mandelbrot_set]({{https://blbadger.github.io}}fractals/mandelbrot_corrected.png)


The Mandelbrot set is a very rich fractal. Here is a zoom on the point - 0.74797 + 0.072500001i (see [here](/julia-sets.md) for a description of how to make the video)

![disappearing mandelbrot]({{https://blbadger.github.io}}fractals/mandelbrot_zoom1.gif)

And here is the same point, increasing scale to a factor o $2^{42}$ (over four trillion)

<iframe width="560" height="315" src="https://www.youtube.com/embed/0qrordbf7WE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

What happens if we change the exponent of (2) such that $z^1 \to z^4$ ?  At $z^1$, the equation is linear and a circular region about the origin remains bounded.  But as the system becomes nonlinear, intricate shapes appear.

![extended mandelbrot]({{https://blbadger.github.io}}fractals/mandelbrot_slow.gif)


What happens if we add a small amount $b$ to $a$?  Then we have $z = z^2 + a + b$, and intuitively one can guess that the bounded set size will decrease as $b$ gets farther from the origin because there is less bounded area far from the origin in the Mandelbrot set. Let's look at many values of a real $b$, going from $b=0 \to b=1.3 \to b=0$:

![disappearing mandelbrot]({{https://blbadger.github.io}}fractals/mandelbrot_disappeared.gif)

In the other direction, $b=0 \to b = -2.5$ yeilds

![disappearing mandelbrot]({{https://blbadger.github.io}}fractals/mandelbrot_disappeared_reversed.gif)


How about if we move to a complex number? The set from $b = 0 \to b = 1 - i$ looks like

![disappearing complex mandelbrot]({{https://blbadger.github.io}}fractals/mandelbrot_complex_disappeared.gif)

Instead of moving from the origin to a given point $b$, let's try rotating about the origin at a radius $r$.  Luckily we are already working with complex numbers so this can be done using the identity

$$
e^{i \pi} = -1
$$

so if we want one complete rotation ($2\pi$ radians) after 300 images (the usual length of the videos on this page),
```python
...
	...

	for i in range(max_iterations):
		# mandelbrot equation
		z_array = z_array**2 + a_array + np.exp(3.1415j * (t/150))
```

which yeilds

![disappearing complex mandelbrot]({{https://blbadger.github.io}}fractals/mandelbrot_swirl_0.3r.gif)

