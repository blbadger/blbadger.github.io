## Semicontinuous Clifford attractor

The clifford attractor, also known as the fractal dream attractor, is the system of equations:


$$
x_{n+1} = sin(ay_n) + c \cdot cos(ax_n) \\
y_{n+1} = sin(bx_n) + d \cdot cos(by_n)
\tag{1}$$


where a, b, c, and d are constants of choice.  It is an attractor because at given values of a, b, c, and d,
any starting point $(x_0, y_0)$ will wind up in the same pattern. 

with $a = 2, b = 2, c = 1, d = -1$ the following map is made:
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_1.png)

with $a = 2, b = 1, c = -0.5, d = -1.01$ , 
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

For $a = 2.1, b = 0.8, c = -0.5, d = -1$, the attractor is three points:
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_0d.png)

For $b = 0.95$, more points:
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_0d2.png)

For $b = 0.981$, the points being to connect to form line segments
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_1d1.png)

And for $b = 0.9818$, a slightly-larger-than 1D attractor is produced
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_1d2.png)

And when $b = 1.7$, a nearly-2d attractor is produced
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/clifford_2d.png)

### Semi-continuous mapping

Say you want to model a continuous differential equation.  If the equation is nonlinear, chances are that there is no analytic solution.  What to do? Do an approximation! Perhaps the simplest way of doing this is by using discrete approximations to estimate where a point will go given its current position and its derivative.  This is known as Euler's method, and can be expressed as follows:
Given 
$
dy / dx = f(x, y), \\
x(0) = y_0
$

$$
x_{n+1} = x_n + dx * \Delta t
$$

at $\Delta t = 0.01$, a smooth path along the vectors is made.  The path is 1D, and the attractor is a point (0D).
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_0.01t.png)

at $\Delta t = 0.1$
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_0.1t.png)

at $\Delta t = 0.8$ (points are connected for clarity)
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_0.8t.png)

$\Delta t = 1.1$
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_1.1t.png)

$\Delta t = 1.15$
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_1.15t.png)

$\Delta t = 1.2$, the first few iterations reveal four slowly rotating lattice points
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_1.2t_lines.png)

with more iterations at *dt* = 1.2, it is clear that the attractor is now 1 dimensional
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_1.2t.png)

$\Delta t = 1.3$
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_1.3t.png)

$\Delta t = 1.35$
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_1.35t_lines.png)

$\Delta t = 1.35$, a shape similar to the discrete map has formed.
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_1.35t.png)

What is the utility of using large $\Deleta t$ values if this is not an accurate

Is this a fractal? Zooming in on the bottom right section suggests that it is:
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_zoom1.png)
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_zoom2.png)
![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_zoom3.png)





