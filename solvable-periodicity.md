### Decidability and periodicity

*This page is under construction*

[Elsewhere](https://blbadger.github.io/aperiodic-irrationals.html) we have seen that the set of all periodic dynamical systems (albeit systems as defined in a specific manner) is equinumerous with the set of solvable problems.  The set of all aperiodic systems was found to be more numerous, instead equivalent to the size of the set of all unsolvable problems.  This page will investigate a direct equivalence between the the members of the sets of solvable problems and periodic maps to arrive at a stronger conclusion: equality (with substitution) between these sets, rather than equivalence.

'Unsolvable' and 'Undecidable' are here defined in terms of mathematical logic: decidability is the process of determining the validity of a $\mathfrak t, \mathfrak f$ statement.  Validity means that the statement is always true, regardless of the inputs to the statement.  Solvable means that there is some computation procedure that applies to the given function (or problem) that is finite in length.

### Decadiability and the Church-Turing thesis

Church and Kleene postulated that all effectively computable functions are $\lambda$-definable, and later shown that these functions are equivalent to the general recursive functions Godel defined to be computable.  

A particularly poignent implication of the Church-Turing thesis is that most functions are uncomputable by any means.  A proof for this is as follows, from Kleene's *Mathematical Logic*:

There are an uncountably infinite number of 1-place number theoretic functions on a countably infinite number of inputs.  Suppose that we could enumerate all such functions in a list $f_0(a), f_1(a), f_2(a), ...$.  Each function has the domain of the natural numbers $0, 1, 2, 3, ...$ so we can list them as follows:

$$
f_0(a): f_0(0), \; f_0(1), \; f_0(2), ...
f_1(a): f_1(0), \; f_1(1), \; f_1(2), ...
f_2(a): f_2(0), \; f_2(1), \; f_2(2), ...
...
$$

Now define a function $f_n(a) = f_a(a) + 1$.  This function is an application of Cantor's diagonal method: it differs from $f_0(a)$ because $f_n(0) = f_0(0) + 1$ and no number is its own successor.  Similarly, it differs from the other functions listed because $f_n(1) = f_1(1) + 1$ and $f_n(2) = f_2(2) + 1$.  No matter how many functions we enumerate, $f_n(a)$ will be different from them all.  But as $f_n(a)$ is a one-place number theoretic function, we have a contradiction and conclude that there is no enumeration of all 1-place number theoretic functions. 

Now note that there are only countably infinite Turing machines, as each machine is defined by a finite list of instructions and these lists can be enumerated one by one.  Therefore there are many fewer Turing machines than possible number-theoretic functions, and as each Turing machine describes one number-theoretic function then necessarily most functions are uncomputable on Turing machines (or equivalently in general recursive functions or by $\lambda$ calculus).  If we accept the Church-Turing thesis, this means that most functions are not computable by any means.  'Most' here is a bit of an understatement, as actually only an infinitely small subset of number theoretic functions are computable.

### Unsolvability and aperiodicity

Solvable problems are those for which, by the Church-Turing thesis, there exists an algorithm of finite size (for a Turing machine) that yields a decision for any of a countably infinite number of decision inputs.  

Take the mapping of Turing machine (or $\lambda$ -recursive function, as in Church's original thesis) inputs to the outputs as a dynamical system, where each element of the set of countably infinite inputs $\{ i_0, i_1, i_2 ... \}$ is mapped to its output $ \{ O_0, O_1, O_2 ... \}$ as 

$$
\{i_0 \to O_0, \; i_1 \to O_1, \; i_2 \to O_2 ...\}
$$

To be solvable, an algorithm of finite length must be able to map all of the countably infinite inputs.  After a certain number of $n$ inputs, the decision procedure for $i_0$ must be identical to the decision procedure $\mathscr P$ for $i_{0+n}$ or else the algorithm would be of infinite size, meaning that it would be unusable as it would never halt when applied to a Turing machine.  Thus $\mathscr P$ repeats after a finite interval $n$ between inputs.  

As the inputs are countable, they can be arranged in sequence.  Taking the sequence of decision procedures $\mathscr D$ for all inputs, the function mapping an input $\{ i_0, i_1, i_2 ... \}$ to its decision procedure $ \{ \mathscr D_0, \mathscr D_1, \mathscr D_2 \}$ defined as $f(i)$ can be viewed as a discrete dynamical system.  For all decidable problems, this function is periodic for some ordering of $i$.

### What does it mean that most problems are unsolvable, or that most $\mathfrak t, \mathfrak f$ predicates are undecidable?

The appearance of undecidable predicates and unsolvable problems (and unprovable but true statements) in arithemtic with the natural numbers is disconcerting enough when one considers how intuitively consistent the axioms of this system are as compared to, say, those of analysis.

On the other hand, consider this: in the real world, many problems appear to be so difficult to find a solution to as to be, for all intents and purposes, impossible.  The study of mathematics attempts to understand the behavior of often very abstract ideas, but it is rooted in a study of the natural world: of shapes made by objects here on earth, paths that stars take in the sky, and the like.  Great utility of mathematics in describing the natural world remains to this day.  

Thus mathematics would be expected to reflect something about the natural world.  Now if all mathematical problems were solvable, it would not seem that this study was an accurate description of the world around us.  If a system with axioms as simple as arithmetic could have unsolvable problems, there is hope that even something as historically removed from application as number theory reflects something very important about 

### Addition-only or multiplication-only number theories are decidable

### Examples of probably-undecidable number theoretic statements

As both Presburger and Skolem arithmetics are decidable (and complete) but arithemtic with both addition and multiplication is not, one would expect to find some undecidable number-theoretic statements existing at the intersection of multiplication and addition.  If we consider prime numbers to the the 'atoms' of multiplication, this is indeed the case: there are a number of as-yet-undecidable statements relating primes to addition. First and most famous is Goldbach's conjecture,

$$
\forall n>2, \; \exists a, b \in \{primes\} : n = a + b
$$

There is also the twin primes conjecture, that there is an infinite number of pairs of primes that are two apart from each other,

$$
\lvert \{p, p+2\} \rvert > n \; \forall n : p, p+2 \in \{ primes \}
$$

and Legendre's conjecture,

$$
\forall n, \; \exists p : n^2 < p < (n+1)^2
$$

All are as yet unproved and yet seem true, at least they are true for all inputs observed thus far.

