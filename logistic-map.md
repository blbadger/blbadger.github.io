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

and at r=3.7, the period is longer than the iterations plotted (actually it is infinite)
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/logistic_time_r3.7.png)


This information may be compiled in what is called an orbit map, which displays the stable points at each value of r.  These may also be though of as the roots of the equation with specific r values.  
On the x-axis is r, on the y-axis is the stable point value:
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/logistic_period.png)







