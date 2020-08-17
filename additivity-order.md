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

As we go from 100 tries of 100 tosses each to 1000000 tries of 100 tosses each, the normal distribution is fit more and more perfectly.  For $n=100, t=100$,the distribution more closely approximates the normal curve.

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

 In three dimensions, Browniam motion leads to a movement away from the initial particle position.  The direction of this movement is unpredictable, but over many experiments (or with many particles underging brownian motion at once), the distances of the particles away from the initial point together form a Gaussian distribution (see [here](https://en.wikipedia.org/wiki/Brownian_motion) for a good summary of this).  
 

### White and Brown noise

White noise is defined as being frequency-independent: over time, neither low nor high frequencies are more likely to be observed than the other.  Brownian noise is defined here as any noise that is inversly frequency dependent: lower frequency signal occurs more often than higher frequency signal.  Inverse frequency noise is also called pink or $1/f$ noise, and here we are calling Brown noise any noise that has $1/f^n$ for $n > 0$. 


### Aside: quantum mechanics and non-differentiable motion
 
The wave-like behavior of small particles such as photons or electrons is one of the fundamnetal aspects of the physics of small objects.  Accordingly, the equations of quantum mechanics have their root in equations describing macroscopic waves of water and sound.  Now macroscopic waves are the result of motion of particles much smaller than the waves themselves.  As quantum particles are well-described by wave equations, it seems logical to ask whether or not these particles are actually composed of many smaller particles in the same way macroscopic waves are.  

As a specific example of what this would mean, imagine a photon as a collection of particles undergoing browninan motion for a time that is manifested by the photon's wavelength: longer wavelengths mean more time has elapsed whereas smaller wavelengths signify less time. When a photon is detected, it forms a Gaussian distribution and this is exactly the pattern formed by a large collection of particles undergoing Brownian motion for a specified time.  The same is true for any small particle: the wavefunctions ascribed to these objects may be represented as Brownian motion of many particles.  

If this seems strange, consider that this line of reasoning predicts that the paths of small particles, as far as they can be determined, are non-differentiable everywhere, just as the path of a particle undergoing Brownian motion is.  In the path integral approach to quantum events, this is exactly what is assumed and the mathematics used to make sense of quantum particle interactions (uncertainty relations etc.) is derived from the mathematics used to understand Brownian motion of macroscopic particles.  This math is tricky because nowhere-differentiable motion is not only poorly understood in a simple form by calculus, but the paths themselves of such particles are infinitely long, leading to a number of paradoxes. 

The similarities that exist between how quantum particles behave and how macroscopic particles behave when undergoing Brownian motion suggests that we consider the possibility of small particles existing in a similar environment to macroscopic ones in fluid.  








 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
