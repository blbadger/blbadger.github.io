## Logistic Map

The logistic equation was derived from a differential equation describing population growth. The equation is as follows:

```python
x_next = r * x_current (1 - x_current)
```

where r can be considered akin to a growth rate, x_next is the population next year, and x_current is the current population.

When r is small (< 1), the population heads towards 0:
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/logistic_time_r0.8.png)

As r is increased past 1, a stable population is reached:
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/logistic_time_r2.5.png)

As r further increases, the population fluctuates, returning to the starting point every other year.  This is called 'period 2':
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/logistic_time_r3.1.png)

at r = 3.5, the populaiton is period 4:
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/logistic_time_r3.5.png)

and at r=3.55, the population is period 8:
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/logistic_time_r3.55.png)

and at r=3.7, the period is longer than the iterations plotted (actually it is infinite)
![t=0.05 map]({{https://blbadger.github.io}}/logistic_map/logistic_time_r3.7.png)

