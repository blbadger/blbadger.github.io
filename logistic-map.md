## Logistic Map

The logistic equation was derived from a differential equation describing population growth. The equation is as follows:

```python
x_next = r * x_current (1 - x_current)
```

where r can be considered akin to a growth rate, x_next is the population next year, and x_current is the current population.

When r is small (< 1), the population heads towards 0:
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/logistic_time_r0.8.png)

As r is increased to 2.5, a stable population is reached:
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/logistic_time_r2.5.png)

If r = 3.1, the population fluctuates, returning to the starting point every other year.  This is called 'period 2':
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/logistic_time_r3.1.png)

at r = 3.5, the population is period 4, as it takes 4 iterations for the population to return to its original position:
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/logistic_time_r3.5.png)

and at r=3.55, the population is period 8:
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/logistic_time_r3.55.png)

and at r=3.7, the period is longer than the iterations plotted (actually it is infinite).  The ensuing plot has points that look random but are deterministic.  The formation of aperiodic, unpredictable behavior from a deterministic system is called mathematical chaos.
![map]({{https://blbadger.github.io}}/logistic_map/logistic_time_r3.7.png)


This information may be compiled in what is called an orbit map, which displays the stable points at each value of r.  These may also be though of as the roots of the equation with specific r values. 

On the x-axis is r, on the y-axis is the stable point value. By looking at how many points there are at a given r value, the same patter of period doubling may be observed. The phenomenon that periodic nonlinear systems become aperiodic via period doubling at specific ratios was found by Feigenbaum to be a near-universal feature of the transition from periodicity to chaos.
![map]({{https://blbadger.github.io}}/logistic_map/logistic_period.png)


### The relationship between chaotic systems and fractals

Let's take a closer look at the fuzzy region of the right. This corresponds to the values of r which are mostly aperiodic, but with windows of periodicity.  There are all kinds of interesting shapes visible!  Thus this chaotic system is not    
![map]({{https://blbadger.github.io}}/logistic_map/logistic_period_zoom2.png)

One of the most striking features of this map is that it is a self-similar fractal.  This means that smaller parts resemble the whole object.  Observe what happens when we zoom in on the upper left section: A smaller copy of the original image is found!
![map]({{https://blbadger.github.io}}/logistic_map/logistic_period_zoom3.png)

If we take this image and again zoom in on the upper left hand corner, again we see the original image!
![map]({{https://blbadger.github.io}}/logistic_map/logistic_period_zoom4.png)

And so on ad infinitum.  An infinite number of smaller images of the original are found nested in the tiny logistic equation!  
Far from being unusual, the formation of fractals from nonlinear systems is the norm provided that they are dissipative and do not explode towards infinity.

What does this mean with regards to chaotic systems? Impossible to solve, but in certain respects scale invariant such that the same patterns reappear at different scales. 


### A logistic map from the Henon attractor

The Henon map equation describes a simplified Poincare section of the Lorenz attractor:
```python
dx = 1 - a * x ** 2 + y
dy = b * x

# in this case,
a =  0.1 
b =  0.03
```

If iterated using Euler's formula with *dt* = 0.047, the following map is produced:
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/henon_logistic.jpg)

It looks like the bifurcation diagram from the logistic attractor! Closer inspection on the chaotic portion reveals an inverted Logistic map.
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/henon_logistic_zoom.png)









