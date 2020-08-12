## Additivity and order

This work will focus on the effects of the additive transformation on a variety of subspaces, from probability to probabilistic signal (noise) to organization. 

### Probability

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

Let's simulate this so that we don't have to actually toss a coin millions of times:

```python
import numpy
import matplotlib.pyplot as plt
from scipy.stats import norm

flips = 100
trials = 1000000
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

For 10 tries of 10 tosses each, the normal distribution is inaccurate.

As we go from 1000 tries of 100 tosses each to 1000000 tries of 100 tosses each, the normal distribution is fit more and more perfectly

**insert here**

Let's take a moment to appreciate what has happened here: a completely random input can be mapped to a curve with arbitrary precision simply by adding outputs together.

### Signal and noise



### White and Brown noise

White noise is defined as being frequency-independent: over time, neither low nor high frequencies are more likely to be observed than the other.  
   


### Brownian motion converts heat to work, albeit briefly

What about at scales more relavant to our existance: does an increase in entropy over time always occur?  At a scale slightly smaller than our own, there is indeed. 

The chaotic motion of small particles in fluid is called Brownian motion, after the naturalist who identified this as a process independant of life.  Brownian motion is the result of the summation of thermal motions of many molecules of liquid pushing against a larger particle. These molecular motions cause the particle to move through the surrounding liquid in a 

Over brief period of time and in a small scale, brownian motion is a conversion of heat energy to work, or equivalently the conversion of undirected motion to directed motion.  
