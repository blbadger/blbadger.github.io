### Solvability and periodicity

[Elsewhere](https://blbadger.github.io/aperiodic-irrationals.html) we have seen that the set of all periodic dynamical systems (albeit systems as defined in a specific manner) is equinumerous with the set of solvable problems.  The set of all aperiodic systems was found to be more numerous, instead equivalent to the size of the set of all unsolvable problems.  This page will investigate a direct equivalence between the the members of the sets of solvable problems and periodic maps to arrive at a stronger conclusion: equality (with substitution) between these sets, rather than equivalence.

'Unsolvable' and 'Undecidable' are here defined in terms of mathematical logic: decidability is the process of determining the validity of a $\mathfrak t, \mathfrak f$ statement.  Validity means that the statement is always true, regardless of the inputs to the statement.  

### Decadiability and the Church-Turing thesis

A particularly poignent implication of the Church-Turing thesis is that most functions are uncomputable by any means.  By this is signified...

### Unsolvability and aperiodicity

Solvable problems are those for which, by the Church-Turing thesis, there exists an algorithm of finite size (for a Turing machine) that yields a decision for any of a countably infinite number of decision inputs.  

Take the mapping of Turing machine (or $\lambda$ -recursive function, as in Church's original thesis) inputs to the outputs as a dynamical system, where each element of the set of countably infinite inputs $\{ i_0, i_1, i_2 ... \}$ is mapped to its output $ \{ O_0, O_1, O_2 ... \}$ as 

$$
\{i_0 \to O_0, \; i_1 \to O_1, \; i_2 \to O_2 ...\}
$$

To be solvable, an algorithm of finite length must be able to map all of the countably infinite inputs.  After a certain number of $n$ inputs, the decision procedure for $i_0$ must be identical to the decision procedure $\mathscr P$ for $i_{0+n}$ or else the algorithm would be of infinite size, meaning that it would be unusable as it would never halt when applied to a Turing machine.  Thus $\mathscr P$ repeats after a finite interval $n$ between inputs.  

As the inputs are countable, they can be arranged in sequence.  Taking the sequence of decision procedures $\mathscr D$ for all inputs, the function mapping an input $\{ i_0, i_1, i_2 ... \}$ to its decision procedure $ \{ \mathscr D_0, \mathscr D_1, \mathscr D_2$ defined as $f(i) \} $ can be viewed as a discrete dynamical system.  For all decidable problems, this functions is periodic for soome ordering of $i$.

### What does it mean that most problems are unsolvable, or that most $\mathfrak t, \mathfrak f$ predicates are undecidable?

The appearance of undecidable predicates and unsolvable problems (and unprovable but true statements) in arithemtic with the natural numbers is disconcerting enough when one considers how intuitively consistent the axioms of this system are as compared to, say, those of analysis.

On the other hand, consider this: in the real world, many problems appear to be so difficult to find a solution to as to be, for all intents and purposes, impossible.  The study of mathematics attempts to understand the behavior of often very abstract ideas, but it is rooted in a study of the natural world: of shapes made by objects here on earth, paths that stars take in the sky, and the like.  Great utility of mathematics in describing the natural world remains to this day.  

Thus mathematics would be expected to reflect something about the natural world.  Now if all mathematical problems were solvable, it would not seem that this study was an accurate description of the world around us.  If a system with axioms as simple as arithmetic could have unsolvable problems, there is hope that even something as historically removed from application as number theory reflects something very important about 

### Addition-only or multiplication-only arithmetics are decidable

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

