
 <head>

  <!-- Global site tag (gtag.js) - Google Analytics -->
  
 <!--
 <script async src="https://www.googletagmanager.com/gtag/js?id=UA-171312398-1"></script>
 <script>
   window.dataLayer = window.dataLayer || [];
   function gtag(){dataLayer.push(arguments);}
   gtag('js', new Date());

   gtag('config', 'UA-171312398-1');
 </script>
 -->

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

![Grid map image]({{https://blbadger.github.io}}misc_images/grid_map_cover.gif)

### [Pendulum phase space](/pendulum-map.md)

![pendulum]({{https://blbadger.github.io}}misc_images/pendulum_cover.png)

## Boundaries 
Trajectories of any dynamical equation may stay bounded or else diverge towards infinity.  The borders between bounded and unbounded trajectories can take on spectacular fractal geometries.  

### [Polynomial roots I](/polynomial-roots.md)

![roots]({{https://blbadger.github.io}}misc_images/newton_cover.png)


### [Polynomial roots II](/polynomial-roots2.md)

![convergence]({{https://blbadger.github.io}}misc_images/newton_boundary_cover.png)


### [Julia sets](/julia-sets.md)

![julia set1]({{https://blbadger.github.io}}misc_images/julia_cover.png)

### [Mandelbrot set](/mandelbrot-set.md)

![disappearing complex mandelbrot]({{https://blbadger.github.io}}fractals/mandelbrot_complex_disappeared.gif)


### [Henon map](/henon-map.md)

![map]({{https://blbadger.github.io}}misc_images/henon_cover.png)


### [Clifford map](/clifford-boundary.md)

![clifford]({{https://blbadger.github.io}}misc_images/clifford_cover.png)


### [Logistic map](/logistic-boundary.md)

![logistic map image]({{https://blbadger.github.io}}misc_images/logistic_bound_cover.png)


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


### [Fractal Geometry](/fractal-geometry.md)

![snowflake]({{https://blbadger.github.io}}/misc_images/fractal_cover.png)


## Physics
As for any natural science, an attempt to explain observations and predict future ones using hypothetical statements called theories.  Unlike the case for axiomatic mathematics, such theories are never proven because some future observation may be more accurately accounted for by a different theory.  As many different theories can accurately describe or predict any given set of observations, it is customary to favor the simplest as a result of Occam's razor.  

### [Three Body Problem](/3-body-problem.md)

![3 body image]({{https://blbadger.github.io}}/3_body_problem/3_body_cover.png)

### [Entropy](/entropy.md)

![malachite]({{https://blbadger.github.io}}misc_images/malachite.png)

### [Quantum Mechanics](/quantum-mechanics.md)

$$
P_{12} \neq P_1 + P_2 \\
P_{12} = P_1 + P_2 + 2\sqrt{P_1P_2}cos \delta
$$

## Biology
The study of life, observations of which display many of the features of nonlinear mathematical systems: an attractive state resistant to perturbation, lack of exact repeats, and simple instructions giving rise to intricate shapes and movements.  

### [Genetic Information Problem](/genetic-info-problem.md)

![coral image]({{https://blbadger.github.io}}misc_images/acropora.png)


### [Homeostasis](/homeostasis.md)

![lake image]({{https://blbadger.github.io}}misc_images/lake.png)


## Deep Learning

Machine learning with layered representations.  Originally inspired by efforts to model the animalian nervous system, much work today is of somewhat dubious biological relevance but is extraordinarily potent for a wide range of applications.  For some of these pages and more as academic papers, see [here](https://arxiv.org/search/?searchtype=author&query=Badger%2C+B+L).

### [Image Classification](/neural-networks.md) 

![neural network architecture](/neural_networks/neural_network.png)


### [Input Attribution and Adversarial Examples](/input-adversarials.md)

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/attributions_highres_cropped.gif)


### [Input Generation I: Classifiers](/input-generation.md)

![generated badger](/neural_networks/two_generated_badgers.png)


### [Input Generation II: Vectorization and Latent Space Embedding](/latent_output.md)

![wordnet recovered from imagenet](/neural_networks/nearest_neighbors_animal_embedding.png)


### [Input Generation III: Input Representations](/input-representation.md)

![resnet googlenet transformation](/neural_networks/resnet_vectorized_to_be_googlenet_goldfinch.png)


### [Input Representation I: Depth and Representation Accuracy](/depth-generality.md)

![layer autoencoding](/neural_networks/representation_cover.png)


### [Input Representation II: Vision Transformers](/vision-transformers.md)

![vision transformer layer representations](/neural_networks/vit_cover.png)

### [Language Representation I: Spatial Information](/language-representations.md)

![vision transformer layer representations](/deep-learning/gpt2_features_viz.png)

### [Language Representation II: Sense and Nonsense](/language-representations-inputs.md)

$$
\mathtt{This \; is \; a \; prompt \; sentence} \\ 
\mathtt{channelAvailability \; is \; a \; prompt \; sentence} \\ 
\mathtt{channelAvailability \; millenn \; a \; prompt \; sentence} \\
\dots \\
\mathtt{redessenal \; millenn-+-+DragonMagazine}
$$

### [Language Representation III: Noisy Communication on a Discrete Channel](/language-discreteness.md)

$$
a = \mathtt{The \; sky \; is \; blue.} \\ 
a_g = \mathtt{The \; sky \; is \; blue \lt s \gt}
$$

<html>
<body>
<span style="color:white">
<span style="background-color: #156e6e">The</span><span style="background-color: #746e6e"> wip</span><span style="background-color: #006e6e">ers</span><span style="background-color: #616e6e"> on</span><span style="background-color: #626e6e"> the</span><span style="background-color: #726e6e"> bus</span><span style="background-color: #ff6e6e"> go</span><span style="background-color: #9e6e6e"> sw</span><span style="background-color: #fa6e6e">ish</span><span style="background-color: #966e6e"> sw</span><span style="background-color: #e46e6e">ish</span>
 </span>
 <br> 
 Predicted next token:  sw
 <br>
</body>
</html>

### [Feature Visualization I](/feature-visualization.md)

![features 2](/neural_networks/featuremap_cover2.png)


### [Feature Visualization II: Deep Dream](/deep-dream.md)

![features](/neural_networks/deep_dream_cover.png)


### [Feature Visualization III: Transformers and Mixers](/transformer-features.md)

![features](/deep-learning/transformer_feature_cover.png)


### [Autoencoders](/autoencoder-representation.md)

![autoencoding of landscapes](/deep-learning/autoencoder_cover.png)


### [Diffusion Inversion](/diffusion-inversion.md)

![features](/neural_networks/diffusion_cover.png)


### [Generative Adversarial Networks](/generative-adversarials.md)

![network architecture](/neural_networks/mnist_2latent_fig.png)


### [Normalization and Gradient Stability](/gradient-landscapes.md)

![network architecture](misc_images/gradient_quivercover.png)


### [Small Language Models for Abstract Sequences](/neural-networks3.md)

![network architecture](/neural_networks/nn_including_embeddings.png)


### [Interpreting Sequence Models](/nn_interpretations.md)

![deep learning attributions](/misc_images/attributions_cover.png)


### [Training Memory](/neural-networks2.md)

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


### [About The Author](/about-me.md)



