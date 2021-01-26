
 <head>

  <!-- Global site tag (gtag.js) - Google Analytics -->
 <script async src="https://www.googletagmanager.com/gtag/js?id=UA-171312398-1"></script>
 <script>
   window.dataLayer = window.dataLayer || [];
   function gtag(){dataLayer.push(arguments);}
   gtag('js', new Date());

   gtag('config', 'UA-171312398-1');
 </script>

 <meta name="Description" CONTENT="Author: Benjamin Badger, Category: Informational">
 <meta name="google-site-verification" content="UtBQXaaKqY6KYEk1SldtSO5XVEy9SmoUfqJ5as0603Y" />
 </head>

Come and see some beautiful mathematical ideas, explored computationally and theoretically.  Follow the links to skip to pages on [dynamical boudaries](#boundaries) or abstract [foundations](#foundations), or related ideas in [physics](#physics) or [biology](#biology).  And just for fun, see some [smaller software projects](#smaller-software-projects) and [high voltage engineering projects](#high-voltage).

## Dynamics
The mathematical approach to change over time. Most dynamical systems are nonlinear, which are not additive, meaning that the sum of an ensemble is more than the parts that compose it.  Nonlinear systems are mostly unsolvable, and though deterministic are unpredictable.  

### [Logistic map](/logistic-map.md)

![logistic map image]({{https://blbadger.github.io}}/misc_images/logistic_zoom.jpg)


### [Clifford attractor](/clifford-attractor.md)

![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semiclifford_vid1.gif)


### [Grid map](/grid-map.md)

![Grid map image]({{https://blbadger.github.io}}grid_map/grid_vid.gif)


### [Pendulum phase space](/pendulum-map.md)

![pendulum]({{https://blbadger.github.io}}pendulum_map/pendulum_cover2.jpg)

### [Polynomial roots](/polynomial-roots.md)

![roots]({{https://blbadger.github.io}}newton-method/Newton046.png)


## Boundaries 
Trajectories of any dynamical equation may stay bounded or else diverge towards infinity.  The borders where bounded and unbounded trajectories exist adjacent to one another can take on spectacular fractal geometries.  

### [Julia sets](/julia-sets.md)

![julia set1]({{https://blbadger.github.io}}fractals/Julia_set_inverted.png)


### [Mandelbrot set](/mandelbrot-set.md)

![disappearing complex mandelbrot]({{https://blbadger.github.io}}fractals/mandelbrot_complex_disappeared.gif)


### [Henon map](/henon-map.md)

![map]({{https://blbadger.github.io}}/henon_map/henon_cover2.png)


### [Clifford map](/clifford-boundary.md)

![clifford]({{https://blbadger.github.io}}clifford_attractor/Clifford_boundary_still.png)


### [Logistic map](/logistic-boundary.md)

![logistic map image]({{https://blbadger.github.io}}/logistic_map/logistic_bound_cover.png)


## Foundations
Why nonlinear systems behave the way they do, in general terms.

### [Aperiodicity implies sensitivity to initial conditions](/chaotic-sensitivity.md)

$$
f(x) : f^n(x(0)) \neq f^k(x(0)) \implies \\
\forall x_1, x_2 : \lvert x_1 - x_2 \rvert < \varepsilon, \; \exists n \; : \lvert f^n(x_1) - f^n(x_2) \rvert > \varepsilon
$$

### [Aperiodic maps, irrational numbers, and solvable problems](/aperiodic-irrationals.md)

$$  
\Bbb R - \Bbb Q \sim \{f(x) : f^n(x(0)) \neq f^k(x(0))\} \\
\text{given} \; n, k \in \Bbb N \; \text{and} \; k \neq n \\
$$

### [Irrational numbers on the real line](/irrational-dimension.md)

$$
\Bbb R \neq \{ ... x \in \Bbb Q, \; y \in \Bbb I, \; z \in \Bbb Q ... \}
$$

### [Discontinuous aperiodic maps](/most-discontinuous.md)

$$
\{ f_{continuous} \} \sim \Bbb R \\
\{ f \} \sim 2^{\Bbb R}
$$

### [Poincare-Bendixson and dimension](/continuity-poincare.md)

$$
D=2 \implies \\
\forall f\in \{f_c\} \; \exists n, k: f^n(x) = f^k(x) \; if \; n \neq k
$$

### [Decidability](/solvable-periodicity.md)

$$
\{i_0 \to O_0, i_1 \to O_1, i_2 \to O_2 ...\}
$$

### [Nonlinearity and dimension](/nonlinear-dimension.md)

![mapping]({{https://blbadger.github.io}}misc_images/curve_mapping.png)

### [Reversibility and periodicity](/aperiodic-inverted.md)

$$
x_{n+1} = rx_n(1-x_n) \\
\; \\
x_{n} = \frac{r \pm \sqrt{r^2-4rx_{n+1}}}{2r}
$$

### [Additive transformations](/additivity-order.md)

![random fractal]({{https://blbadger.github.io}}/misc_images/randomized_sierpinksi_2.gif)


### [Fractal geometry](/fractal-geometry.md)

![snowflake]({{https://blbadger.github.io}}/fractals/snowflake_fractal.gif)


## Physics
As for any science, an attempt to explain observations and predict future ones using hypothetical theories.  Unlike the case for axiomatic mathematics, such theories are never proven because some future observation may be more accurately described by a different theory.  As many different theories can accurately describe and predict the same observations, the simplest are favored by Occam's razor.  

### [Entropy](/entropy.md)

![malachite]({{https://blbadger.github.io}}/assets/images/malachite.png)

### [3 body problem](/3-body-problem.md)

![3 body image]({{https://blbadger.github.io}}/3_body_problem/3_body_cover.png)

### [Quantum mechanics](/quantum-mechanics.md)

$$
P_{12} \neq P_1 + P_2
$$

## Biology
The study of life, which displays many of the features of nonlinear mathematical systems: an attractive state resistance to perturbation, lack of periodicity, and simple instructions giving rise to intricate shapes and movements.  

### [Genetic information problem](/genetic-info-problem.md)

![coral image]({{https://blbadger.github.io}}/bio_images/acropora.png)


### [Homeostasis](/homeostasis.md)

![lake image]({{https://blbadger.github.io}}/bio_images/lake.png)


### [Neural Network](/neural-networks.md) 

![neural network architecture]({{https://blbadger.github.io}}/misc_images/cNN_architecture.png)


## Smaller software projects

### [Game puzzles](/puzzle-projects.md)

![puzzles]({{https://blbadger.github.io}}/assets/images/games.png)

### [Programs to compute things](/computing-programs.md)

$$
\; \\
\begin{vmatrix}
a_{00} & a_{01} & a_{02} & \cdots & a_{0n} \\
a_{10} & a_{11} & a_{12} & \cdots & a_{1n} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
a_{n0} & a_{n1} & a_{n2} & \cdots & a_{nn} \\
\end{vmatrix}
\; \\
$$

$$ 
\; \\
5!_{10} = 12\mathbf{0} \to 1 \\
20!_{10} = 243290200817664\mathbf{0000} \to 4 \\
n!_k \to ?
\; \\
$$
 	
## High voltage 
High voltage engineering projects: follow the links for more on arcs and plasma.

### [Tesla coil](/tesla-coils.md)

![tesla coil arcs]({{https://blbadger.github.io}}tesla_images/newtesla.jpg)


### [Fusor](/fusor.md)

![fusor image]({{https://blbadger.github.io}}fusor_images/fusor-1-1.png)


### [About Me](/about-me.md)



