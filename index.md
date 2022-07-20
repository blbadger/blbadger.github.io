
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

## Dynamics
The mathematical approach to change over time. Most dynamical systems are nonlinear and generally unsolvable, and though deterministic are often unpredictable.  

### [Logistic map](/logistic-map.md)

![logistic map image]({{https://blbadger.github.io}}logistic_map/logistic_trace0428_cropped.png)


### [Clifford attractor](/clifford-attractor.md)

![clifford vectors image]({{https://blbadger.github.io}}clifford_attractor/semi_clifford_cover.png)


### [Grid map](/grid-map.md)

![Grid map image]({{https://blbadger.github.io}}grid_map/grid_vid.gif)


### [Pendulum phase space](/pendulum-map.md)

![pendulum]({{https://blbadger.github.io}}pendulum_map/pendulum_cover2.jpg)

## Boundaries 
Trajectories of any dynamical equation may stay bounded or else diverge towards infinity.  The borders between bounded and unbounded trajectories can take on spectacular fractal geometries.  

### [Polynomial roots I](/polynomial-roots.md)

![roots]({{https://blbadger.github.io}}newton-method/Newton046.png)

### [Polynomial roots II](/polynomial-roots2.md)

![convergence]({{https://blbadger.github.io}}/newton-method/Newton_boundaryx5-x-1.png)


### [Julia sets](/julia-sets.md)

![julia set1]({{https://blbadger.github.io}}fractals/Julia_set_inverted.png)

### [Mandelbrot set](/mandelbrot-set.md)

![disappearing complex mandelbrot]({{https://blbadger.github.io}}fractals/mandelbrot_complex_disappeared.gif)


### [Henon map](/henon-map.md)

![map]({{https://blbadger.github.io}}/henon_map/henon_cover2.png)


### [Clifford map](/clifford-boundary.md)

![clifford]({{https://blbadger.github.io}}clifford_attractor/clifford_cover0.png)


### [Logistic map](/logistic-boundary.md)

![logistic map image]({{https://blbadger.github.io}}/logistic_map/logistic_bound_cover.png)


## Foundations

### [Primes are unpredictable](/unpredictable-primes.md) 

$$
\lnot \exists n, m : (g_n, g_{n+1}, g_{n+2}, ... , g_{n + m - 1}) \\
= (g_{n+m}, g_{n+m+1}, g_{n+m+2}, ..., g_{n + 2m - 1}) \\
= (g_{n+2m}, g_{n+2m+1}, g_{n+2m+2}, ..., g_{n + 3m - 1}) \\
\; \; \vdots
$$

### [Aperiodicity implies sensitivity to initial conditions](/chaotic-sensitivity.md)

$$
f(x) : f^n(x(0)) \neq f^{n+k}(x(0)) \forall k \implies \\
\forall x_1, x_2 : \lvert x_1 - x_2 \rvert < \epsilon, \; \\
\exists n \; : \lvert f^n(x_1) - f^n(x_2) \rvert > \epsilon
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

### [Poincaré-Bendixson and dimension](/continuity-poincare.md)

$$
D=2 \implies \\
\forall f\in \{f_c\} \; \exists n, k: f^n(x) = f^k(x) \; if \; n \neq k
$$

### [Computability and Periodicity I: the Church-Turing thesis](/solvable-periodicity.md)

$$
\\
\{i_0 \to O_0, i_1 \to O_1, i_2 \to O_2 ...\}
$$

### [Computability and Periodicity II](/uncomputable-aperiodics.md)

$$
x_{n+1} = 4x_n(1-x_n) \implies \\
x_n = \sin^2(\pi 2^n \theta) 
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

![random fractal]({{https://blbadger.github.io}}/misc_images/additivity_cover.png)


### [Fractal geometry](/fractal-geometry.md)

![snowflake]({{https://blbadger.github.io}}/misc_images/fractal_cover.png)


## Physics
As for any natural science, an attempt to explain observations and predict future ones using hypothetical statements called theories.  Unlike the case for axiomatic mathematics, such theories are never proven because some future observation may be more accurately accounted for by a different theory.  As many different theories can accurately describe or predict any given set of observations, it is customary to favor the simplest as a result of Occam's razor.  

### [3 body problem](/3-body-problem.md)

![3 body image]({{https://blbadger.github.io}}/3_body_problem/3_body_cover.png)

### [Entropy](/entropy.md)

![malachite]({{https://blbadger.github.io}}/assets/images/malachite.png)

### [Quantum mechanics](/quantum-mechanics.md)

$$
P_{12} \neq P_1 + P_2 \\
P_{12} = P_1 + P_2 + 2\sqrt{P_1P_2}cos \delta
$$

## Biology
The study of life, observations of which display many of the features of nonlinear mathematical systems: an attractive state resistant to perturbation, lack of exact repeats, and simple instructions giving rise to intricate shapes and movements.  

### [Genetic information problem](/genetic-info-problem.md)

![coral image]({{https://blbadger.github.io}}/bio_images/acropora.png)


### [Homeostasis](/homeostasis.md)

![lake image]({{https://blbadger.github.io}}/bio_images/lake.png)


## Deep Learning

### [Image Classification](/neural-networks.md) 

![neural network architecture](/neural_networks/neural_network.png)


### [Input Attribution and Adversarial Examples](/input-adversarials.md)

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/attributions_highres_cropped.gif)


### [Input Generation I](/input-generation.md)

![generated badger](/neural_networks/two_generated_badgers.png)


### [Input Generation II: Vectorization and Latent Space Embedding](/latent_output.md)

![generated badger](/neural_networks/nearest_neighbors_animal_embedding.png)


### [Input Generation III: Autoencoding with Classification Models](/input-autoencoders.md)

![generated badger](/neural_networks/resnet_vectorized_to_be_googlenet_goldfinch.png)


### [Feature Visualization I](/feature-visualization.md)

![features](/neural_networks/featuremap_cover2.png)


### [Feature Visualization II: Deep Dream](/deep-dream.md)

![features](/neural_networks/deep_dream_cover.png)


### [Generative Adversarial Networks](/generative-adversarials.md)

![network architecture](/neural_networks/mnist_2latent_fig.png)


### [Normalization and Gradient Stability](/gradient-landscapes.md)

![network architecture](/neural_networks/gradient_quivercover.png)


### [Small Language Models for Abstract Sequences](/neural-networks3.md)

![network architecture](/neural_networks/nn_including_embeddings.png)


### [Interpreting Sequence Models](/nn_interpretations.md)

![deep learning attributions](/neural_networks/attributions_cover.png)


### [Training Memory](/neural-networks2.md)

$$
v_0 + i_0 \to o_0 \to v_1 \\
v_1 + i_1 \to o_1 \to v_2 \\
\vdots \\
v_{n-1} + i_n \to o_n \to v_n
$$


### [Limitations of Neural Networks](/nn-limitations.md)

![discontinous proof]({{https://blbadger.github.io}}/neural_networks/discontinous_proof.png)


## Small Projects

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



