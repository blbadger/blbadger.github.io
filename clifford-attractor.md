## Semicontinuous Clifford attractor

The clifford attractor, also known as the fractal dream attractor, is the system of equations:
```python3
	x_next = np.sin(a*y) + c*np.cos(a*x) 
	y_next = np.sin(b*x) + d*np.cos(b*y)
```
where a, b, c, and d are constants of choice.  It is an attractor because at given values of a, b, c, and d,
any starting point will wind up in the same pattern. 

with a = 2, b = 2, c = 1, d = -1 the following map is made:
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_1.png)

with a = 2, b = 1, c = -0.5, d = -1.01, 
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_2.png)

A wide array of shapes may be made by changing the constants, and the code to do this is 
[here](https://github.com/blbadger/2D_strange_attractors/blob/master/clifford_attractor.py)
Try experimenting with different constants!

### Clifford attractors are fractal

In what way is this a fractal?  Let's zoom in on the lower right side: 
In the central part that looks like Saturn's rings, there appear to be 6 lines.
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_zoom1.png)

At a smaller scale, however, there are more visible, around 10
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_zoom2.png)

Is this all? Zooming in once, I count 14 lines, with 7 on the top section
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_zoom3.png)

Zooming in on the top shows that there are actually more that 14 lines!
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_zoom4.png)

To drive the point home, the top-right line looks to be as solid a line as any, but on closer inspection it is not. 
There are two paths visible at higher magnification:
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_zoom5.png)

### Clifford attractors can shift between one and two (or more) dimensions

For a = 2.1, b = 0.8, c = -0.5, d = -1, the attractor is three points:
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_0d.png)

For b = 0.95, more points:
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_0d2.png)

For b = 0.981, the points being to connect to form line segments
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_1d1.png)

And for b = 0.9818, a slightly-larger-than 1D attractor is produced
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_1d2.png)

And when b = 1.7, a nearly-2d attractor is produced
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_2d.png)

### Semi-continuous mapping

Say you want to 
at *dt* = 0.01, a smooth path along the vectors is made.  The path is 1D, and the attractor is a point (0D).
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_0.01t.png)

at *dt* = 0.1
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_0.1t.png)

at *dt* = 0.8
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_0.8t.png)

*dt* = 1.1
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_1.1t.png)

*dt* = 1.15
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_1.15t.png)





