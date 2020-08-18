## Additivity and order

### Probability and additivity

Imagine tossing a fair coin and trying to predict the outcome. There exists a 50% chance of success, but there is no way to know with any certainty what will happen.  Now imagine tossing the coin thousands of times, and consider what you will be able to predict: each toss continues to be random and completely uncertain, so one's ability to predict the outcome of each individual toss does not change.  But now add the results of the tosses together, assigning an arbitrary value to heads and a different value to tails.  As the number of individual coin tosses increases, one is better able to predict what value the sum will take.  Moreover, if the same coin toss experiment is repeated many times (say repeating 1000 tosses 1000 times), we can predict what the sum of the tosses will be with even more accuracy.  


Specifically, there are two possible outcomes for each toss (heads or tails) and so a binomial distribution is appropriate to model the expected value $E(X)$ after $n$ tosses as follows:

$$
E(X) = n * p
$$

where $p$ is probability of the coin landing in a specific orientation, say heads. 

The variance $V(X)$ approximates a binomial distribution centered around the expected value of each toss times the number of tosses.

$$  
V(X) = n * p (1-p) \\
V(X) = n(1/4)^2
$$

As n increases, the standard deviation shrinks with respect to the expected value: after 100 tosses the standard deviation is 10% of the expected value, and after 1000 tosses the standard deviation is ~3% of the expected value, whereas after 10000 tosses the standard deviation is only 1% of the expected value.  

The second observation is perhaps more striking: the additive transformation on a coin toss leads to the gaussian distribution with arbitrarily close presicion.  Let's simulate this so that we don't have to actually toss a coin millions of times, assigning the value of $1$ to heads and $0$ to tails.  We can compare the resulting distribution of heads to the expected normal distribution using `scipy.stats.norm` as follows:

```python
import numpy
import matplotlib.pyplot as plt
from scipy.stats import norm

flips = 10
trials = 10
sum_array = []
for i in range(trials):
	s = 0
	for j in range(flips):
		s += numpy.random.randint(0, 2)
	sum_array.append(s)

variance = flips * (0.5) ** 2
y = numpy.linspace(0, flips, 100)
z = norm.pdf(y, loc=(flips/2), scale=(variance**0.5))

final_array = [0 for i in range(flips)]
for i in sum_array:
	final_array[i] += 1

fig, ax = plt.subplots()
ax.plot(final_array, alpha=0.5)
ax.plot(y, z*trials, alpha=0.5)
plt.show()
plt.close()
```

Let's assign $n$ as the number of tosses for each experiment and $t$ as the number of experiments plotted.  For 10 experiments ($t=10$) of 10 flips ($n=10$) each, the normal distribution (orange) is an inaccurate estimation of the the actual distribution (blue).

![gaussian]({{https://blbadger.github.io}}/assets/images/coin_10_10.png)

As we go from 100 tries of 100 tosses each to 1000000 tries of 100 tosses each, the normal distribution is fit more and more perfectly.  For $n=100, t=100$,the distribution more closely 
approximates the normal curve.

![gaussian]({{https://blbadger.github.io}}/assets/images/coin_100_100.png)

For $n = 100, t = 1000$, 

![gaussian]({{https://blbadger.github.io}}/assets/images/coin_100_1k.png)

for $n=100, t=1 * 10^4$,

![gaussian]({{https://blbadger.github.io}}/assets/images/coin_100_10k.png)

for $n=100, t=1 * 10^5$,

![gaussian]({{https://blbadger.github.io}}/assets/images/coin_100_100k.png)

and for $n=100, t=1 * 10^6$, there is almost no discernable difference between the Gaussian curve centered on the expectation value of $50$ and the actual distribution.

![gaussian]({{https://blbadger.github.io}}/assets/images/coin_100_1mil.png)


Let's take a moment to appreciate what has happened here: a completely random input can be mapped to a curve with arbitrary precision simply by adding outputs together.  (Well, not completely random: digital computers actuqlly produe pseudo-random outputs that eventually repeat over time. But we do not need to worry about that here, as our output number is far less than what would be necessary for repetition.)

This observation is general: individual random events such as a die roll or card shuffle cut are quite unpredictable, but adding together many random events yields a precise mapping to a gaussian curve centered at the expectation value.  Each individual event remains just as unpredictable as the last, but the sum is predictable to an arbitrary degree given enough events.  One way to look at these observations is to see that addition orders random events into a very non-random map, and if we were to find the sum of sums (ie integrate under the gaussian curve) then a number would be reached with arbitrary precision given unlimited coin flips and experiments.

### Brownian motion

There is an interesting physical manifestation of the abstract statistical property of additive ordering: Brownian motion, the irregular motion of small (pollen grain size or smaller) particles in fluid.  Thermal motions (often thought of as random, or stochastic) of fluid molecules add together over time to result in a non-differentiable path of the particle, and movement along this path is termed Brownian motion, after the naturalist Brown who showed that this motion is not the result of a biological process (although it very much resembles the paths taken by small protists living in pond water). 

In three dimensions, Browniam motion leads to a movement away from the initial particle position.  The direction of this movement is unpredictable, but over many experiments (or with many particles underging brownian motion at once), the distances of the particles away from the initial point together form a Gaussian distribution (see [here](https://en.wikipedia.org/wiki/Brownian_motion) for a good summary of this).  Regardless of the speed or trajectory of each individual particle undergoing Brownian motion an ensemble that start at the same location form a Gaussian distribution with arbitrary accuracy, given enough particles. 

### White and fractional noise

Noise may signify the presence of sound, or it may mean everything that is not a signal. Both meanings are applicable here. White noise is defined as being frequency-independent: over time, neither low nor high frequencies are more likely to be observed than the other.  Fractional noise is defined here as any noise that is inversly frequency dependent: lower frequency signal occurs more often than higher frequency signal.  Inverse frequency noise is also called pink or $1/f$ noise, and Brown noise any noise that has $1/f^n$ for $n > 0$.

White noise is unpredictable, and 'purely' random: one cannot predict the frequency or intensity of a future signal any better than by chance.  But because fractional noise decays in total intensity with an increase in frequency, this type of noise is not completely random (meaning that it is somewhat predictable).  Consider one specific type of fractional noise, Brown noise, which is characterized by a $1/f^{~2}$ frequency vs. intensity spectrum.  This is the noise that results from Brownian motion (hence the name).  It may come as no surprise that another way to generate this noise is to integrate (sum) white noise, as the section above argues that Brownian motion itself acts to integrate random thermal motion.  

As noted by [Mandelbrot](https://books.google.com/books/about/The_Fractal_Geometry_of_Nature.html?id=0R2LkE3N7-oC) and [Bak](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.59.381), fractional noise results from fractal objects as they change over time.  Brownian noise coming from Brownian motion is simply a special case of this more general phenomenon: Brownian motion traces out fractal paths.  These paths are self-similar in that very small sections of the path resemble much larger sections, not that smaller portions exactly match the whole as is the case for certain geometric fractals.  This type of self-similarity is sometmies called statistical self-similarity, and is intimately linked to the property this motion exhibits of being nowhere-differentiable.

[generate Weierstrauss function animation]

In general terms, Brownian motion is a fractal because it appears to trace paths that are rough and jagged at every scale until the atomic, meaning that at large scales the paths somewhat resemble the Weierstrauss function above.  Now imaging trying to measure the length of such a shape: the more you zoom in, the longer the measurement is!  Truly nowhere-differentiable paths are of infinite length and instead can be characterized by how many objects (a box or sphere or any other regular shape) it takes to cover the curve at smaller and smaller scales (either by making the shapes themselves smaller or by making the curve larger).  The ratio of the logarithm of the change in the number of objects necessary for covering the curve (a fraction) divided by the logarithm of the change in scale is the counting dimension.  Equivalently,

$$
D = \frac{log(N)}{log(e)} //
N = e^D
$$

where $D$ is the counting dimension, $N$ is the ratio of the number of objects required to cover the curve at the smaller vs. larger scale, and $e$ is the change in scale.  A line has a topological and counting dimension of 1: it takes twice as many objects to cover it when the scale has increased by a factor of two, and similarly a square in a plane has a topological and counting dimension of 2. 

If the counting dimension is larger than the object's topological dimension, it is termed a fractal.  For very rough paths, an increase in scale by, say, twofold leads to a larger increase in the number of objects necessary to cover the path because the path length has increased relative to the scale. Brownian trails have a counting dimension of approximately $2$, meaning that they practically cover area even though they are topologically one-dimensional. The terms 'fractal dimension' and 'counting dimension' are used here to mean the same thing. 



### Aside: quantum mechanics and non-differentiable motion
 
The wave-like behavior of small particles such as photons or electrons is one of the fundamnetal aspects of the physics of small objects.  Accordingly, the equations of quantum mechanics have their root in equations describing macroscopic waves of water and sound.  Now macroscopic waves are the result of motion of particles much smaller than the waves themselves.  As quantum particles are well-described by wave equations, it seems logical to ask whether or not these particles are actually composed of many smaller particles in the same way macroscopic waves are.  

As a specific example of what this would mean, imagine a photon as a collection of particles undergoing browninan motion for a time that is manifested by the photon's wavelength: longer wavelengths mean more time has elapsed whereas smaller wavelengths signify less time. When a photon is detected, it forms a Gaussian distribution and this is exactly the pattern formed by a large collection of particles undergoing Brownian motion for a specified time.  The same is true for any small particle: the wavefunctions ascribed to these objects may be represented as Brownian motion of many particles.  

If this seems strange, consider that this line of reasoning predicts that the paths of small particles, as far as they can be determined, are non-differentiable everywhere, just as the path of a particle undergoing Brownian motion is.  In the path integral approach to quantum events, this is exactly what is assumed and the mathematics used to make sense of quantum particle interactions (uncertainty relations etc.) is derived from the mathematics used to understand Brownian motion of macroscopic particles.  This math is tricky because nowhere-differentiable motion is not only poorly understood in a simple form by calculus, but the paths themselves of such particles are infinitely long, leading to a number of paradoxes. 

The similarities that exist between how quantum particles behave and how macroscopic particles behave when undergoing Brownian motion suggests that we consider the possibility of small particles existing in a similar environment to macroscopic ones in fluid.  








 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
