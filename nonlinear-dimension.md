## Nonlinear tranformations change attractor dimension

Here 'nonlinear transformations' is taken to mean the set of all equations that are nonlinear (ie any non limited to unary exponents) or piecewise linear.

### Maps from finite to infinite are nonlinear

Topological dimension and scaling (fractal) dimension are equivalent for the classical objects in which topological dimension is defined: a point is 0-dimensional, a line 1-dimensional, a plane surface two-dimensional etc.  An object in one dimension may be

[insert curve mapping line segment to all reals]

### Changes in dimension require infinite maps

see also space-filling curve length

### Curves are not entirely one-dimensional

Here dimension is not fractal or topological, but a definition based on the transformations that can result from a function.  Take an arbitrary line existing on a plane.  In some orientation, this line may be defined entirely on the basis of another line that exists alone, ie in one dimension.  

Now note that this is not the case for an arbitrary curve on a plane: this curve by definition is not congruent to any of the lines mentioned above. The curve may be mapped to a line, but it is not equal in that it cannot be substituted for any line.  Therefore a curve cannot be defined in one dimension, because otherwise it would be congruent with a line.

This idea extends to curved surfaces in three dimensions.  Such surfaces are not entirely planar but exist in three dimensions, and therefore are not congruent with a topologically two-dimensional plane. 

This goes to suggest that curves of any kind exist between topological dimensions, which provides a clear reason why nonlinear transformations are capable of attaining a fractional dimension.

### Aperiodic, bounded maps are discontinuous

This was shown using analytical methods on [this page](/most-discontinuous.md) for dimensions 1 and 2.  Here is another with a different approach:

A bijective mapping from two dimensions to one is necessarily discontinuous (a proof is found in Pierce’s Introduction to Information Theory, and found at the end of [this page](/neural-networks.md)).  


### Dimension-changing maps are undecidable 

Space-filling Hilbert or Peano curves require an infinite number of recursions.  Infinitely recursive functions are undecidable, and therefore space-filling curves are undecidable.  

Dimension-changing maps require an infinite number of recursions as well, and therefore dimension-changing maps are undecidable. For more on decidability, see [here](/solvable-periodicity.md).


### Additivity and infinite subsets

What does it mean that nonlinear transformations are not additive, which in symbols is

$$
f(a+b) \neq f(a) + f(b)
$$

in the sense that how can this come about?  Intuitively, this means that the transformation specified by $f$ is different at different locations, such that the location $a+b$ is a different transformation than that which is found at $a$ or $b$ and that simply performing the transformation at $a$ then at $b$ is not sufficient to predict the transformation at $a+b$.  In sense, the transformations are unique.

But this remains a somewhat counter-intuitive idea. We can define the set of all possible transformations for any given $f$ as $\mathscr S$.  Now as all finite quantities can be added, elements $a$ and $b$ of $\mathscr S$ are not finite.  An infinite set is equivalent to one of its proper subsets, meaning that inside elements $a$ and $b$ there are more of $a$ and $b$.  This perhaps makes it intuitively mmore understandable as to why these elements cannot be added.

Now recall that nonlinear transformations are usually aperiodic and that bounded, aperiodic maps form self-similar fractals. The observations that $a \in a$ (proper subset) gives a clear reason why such objects exist: each fractal image $i$ contains a smaller version of itself, meaning that $i \in i$ (proper subset, as the smaller version does not contain all points of the larger).  

### Infinite recursion in nonlinear transformations

The images of fractals on this site are examples of recursion without a base case.  In the natural numbers with addition, 1 is the base case for which succession acts to make all other numbers.  A recursive base case is the smallest indivisible unit. 

For nonlinear transformations that yield fractal objects, there is no recursive base case because every object is equivalent to a smaller version of itself. 

### Nonlinearity and problem solving

Nonlinear systems are not additive (see above), meaning that they cannot be broken down into parts and re-assembled again.  As we have seen for fractals, attempting to isolate a small portion of a nonlinear map may simply lead to the presence of the entire map in miniature (a property called self-similarity).  

Now this is a serious challenge to any attempt to understand nonlinear systems because so much of human thought (or at least current problem solving thought) operates by the principle of additivity.  For example, if one builds a house then the a foundation is made, the walls go up, and the roof is set.  Or if a computer program is written, each individual variable is defined and operations are enumerated.  The same is true of practically every discipline of knowledge: smaller pieces are added together to make the final outcome.  

In the models of nonlinear equations on this webpage, computer programs were used to represent the behavior of various nonlinear transformations.  These programs work by dividing each problem into smaller versions of the same problem until a finite case is reached.  For example, a program that observed the behavior of a particle over time first divides the time period into maybe 500000 individual time steps and then proceeds to calculate the behavior as if each time step were itself indivisible.  Or perhaps we wish to understand which points in the complex plane diverge to infinity and which do not: in this case a certain resolution for the plane, perhaps 1400x2000, is chosen and then the maximum number of iterations is specified such that the program eventually halts. 

In either case, the program generates (albeit high-resolution) approximations of the actual object.  These objects are necessarily approximations any machine that attempts recursion without a base case will never halt.  This is why Turing's construct of a machine $T(a, a, x)$ never halts, ie the search for a solution to the problem 'Does a Turing machine with the input $a$ as the program it runs on halt at time x?' is undecidable inside the system because it does not halt.









