## Primes are unpredictable

Whilst incrementing through the integers, how long one has to wait until reaching a prime number is often quite a difficult thing to predict.  The question of where the next prime number will be is equivalent to the question of what is the gap between the current prime and the next prime number.  Gaps between prime numbers form a non-repeating sequence, as do the gaps between those gaps.  This makes it difficult to predict how far away the next prime will be without simply computing the primality of each successive integer, either directly or by an interval technique (eg. the sieve of Eratosthenes).  

### Theorem: The sequence of gaps between consecutive prime numbers is not initially periodic

Restated, the sequence between gaps betwen consecutive prime numbers, starting from the first gap $g_1$

$$
g_1 = p_2 - p_1 = 3-2 = 1
$$            

cannot be split into repeated subsequences, such that repetitions of this subsequence cover all prime gaps.  In other words, given a sequence of gaps between consecutive primes $g_1, g_2, g_3 , ... $ there is no way to divide this list into subsequences $ (g_1, g_2, ... g_{n-1}), (g_{n}, g_{n+1}, ... g_{2n-1}), (g_{2n}, g_{2n+1}, ... g_{3n-1}), ... $ such that $g_1 = g_{n} = g_{2n} ..., g_2 = g_{n+1} = g_{2n+1} ...$.  

**Proof:** 

Enumerating the prime numbers, we have

$$
p_1, \; p_2, \; p_3, \; p_4, \; p_5, \; p_6, \; p_7, \; p_8 ...\\
2, \; 3, \; 5, \; 7, \; 11, \; 13, \; 17, \; 19 ...
$$

and the gaps between primes where $g_n = p_{n+1} - p_n$ is listed as

$$
g_1, \; g_2, \; g_3, \; g_4, \; g_5, \; g_6, \; g_7 ...\\
1, \; 2, \; 2, \; 4, \; 2, \; 4, \; 2 ...
$$

Now suppose that this sequence of gaps between primes did repeat for multiples of some finite $n > 0$.  Then

$$
(g_1, g_2, g_3, ..., g_{n-1}) = (g_{n}, g_{n+1}, g_{n+2}, ... , g_{2n-1}) \\
g_1 = g_{n}, \; g_2 = g_{n+1}, \; ... g_{n-1} = g_{2n-1}
$$

Adding the sequence of gaps together, we have a sum $\mathscr S_n$ 

$$
g_1 + g_2 + g_3 + \cdots + g_{n-1} = \mathscr S_n
$$

which is an integer and thus is either even or odd.  If $\mathscr S_n$ is even, prime $p_n$ is necessarily even as well because $p_n = 2 + \mathscr S_n$.  But if $\mathscr S_n$ is odd, then $p_{2n} = 2 + 2(\mathscr S_n) = 2(1 + \mathscr S_n)$ and therefore $p_{2n}$ is even. But this is a contradiction as the only even prime number is $p_1 = 2$ but by definition for any $n > 0$, $p_{2n} > 2$.  Therefore the sequence of gaps between consecutive primes starting from the first prime cannot repeat itself no matter how large $n$ is (remaining finite), or in dynamical terms all subsequences of $g_1, g_2, g_3, ... $ are aperiodic.

### Theorem: Prime gap sequences are not eventually periodic

In symbols, there does not exist finite $n, m$ such that the sequence of prime gaps may at prime $p_n$ and on be split into subsequences of length $m$

$$
\lnot \exists n, m : (g_n, g_{n+1}, g_{n+2}, ... , g_{n + m - 1}) \\
= (g_{n+m}, g_{n+m+1}, g_{n+m+2}, ..., g_{n + 2m - 1}) \\
= (g_{n+2m}, g_{n+2m+1}, g_{n+2m+2}, ..., g_{n + 3m - 1}) \\
\; \; \vdots
$$

**Proof:**

Suppose that there was some finite $n>0$ such that the above statement held.  Then there exists a finite $m>0$ such that after $m$ gaps, the sequence $g_n, g_{n+1}, ...$ is repeated (with period $m$):

$$
(g_n, g_{n+1}, ... , g_{n+m-1}) = \\
(g_{n+m}, g_{n+m+1}, ... , g_{n+2m-1}) = \\
(g_{n+2m}, g_{n+2m+1}, ... , g_{n+3m-1}) = \\
\; \; \vdots
$$

What happens when this pattern repeats $p_n$ times?  The prime number indexed at this position is $p_{n+mp_n}$, because there are $n$ primes before $g_n$, and $mp_n$ primes after: in this enumeration $g_n = p_{n+1} - p_n$ means that $g_{n+mp_n-1} = p_{n + mp_n} - p_{n+mp_n-1}$.   Defining $\mathscr S_n = g_n + g_{n+1} + \cdots + g_{n+m-1}$ as above, the value of $p_{n+mp_n}$ is 

$$
p_{n+mp_n} = p_n + p_n \mathscr S_n \\
p_{n+mp_n} = p_n (1 + \mathscr S_n)
$$

and therefore $p_n \rvert p_{n+mp_n}$ or in words $p_n$ divides $p_{n+mp_n}$.  But then $p_n < p_{n+mp_n}$ for all $n, m>0$, so therefore $p_{n+mp_n}$ is composite, a contradiction as it was ealier stated to be prime. As $n$ and $m$ were chosen arbitrarily, there is no $n$ or $m$ such that the sequence of prime gaps is eventually periodic given that there are infinitely many prime numbers.  

One can appreciate that this is a more general theorem than the first presented on this page, which is the special case where $p_n = 2$.  

### Theorem: Prime gap sequences are aperiodic

**Proof:** As prime gap sequences are neither initially nor eventually periodic for any finite period $m$, they are aperiodic.

### Theorem: The sequence of gaps-of-gaps of primes is aperiodic

Restated, the sequence of gaps $g_g_1, g_g_2, g_g_3, ...$ between prime gaps such that $g_g_1 = g_2 - g_1$ and $g_g_2 = g_3 - g_2$ etc. is aperiodic


**Proof:** Suppose that the sequence $g_g_1, g_g_2, g_g_3, ...$ were periodic with finite periodicity $m$.  

$$
p_1,\; p_2,\; p_3,\; p_4,\; p_5,\; \\
\; g_1, \; g_2,\; g_3,\; g_4,...\\
\;  \; g_g_1,\; g_g_2,\; g_g_3 ...
$$

Bearing in mind that $g_1=1$ and $g_2 = 2$, $g_g_1 = 1 = g_g_m$.  It is apparent that all of $g_2, g_3, g_4...$ are even, because all primes after $p_1 = 2$ are odd.  But then $g_{m+1} = 1 + g_m$ would be odd, and a contradiction has been reached.  Therefore there is no periodicity in the sequence of gaps between prime gaps.

**Alternate Proof:** 

$$
\mathscr S = \theta
$$

$\mathscr S$ cannot be negative if there are infinitely many positive prime gaps.  But $\mathscr S$ cannot be 0 either, because then the sequence of prime gaps $g_0, g_1, g_2...$ would be periodic.  And finally $\mathscr S$ cannot be positive, because then there would be 0 or finitely many instances of any gap size, contradicting Zhang's theorem that there is some $k < 70000000$ for which there are infinitely many prime gaps of size $k$, or Maynard's finding of a $k<600$ with the same properties.  As $\mathscr S$ cannot be neither greater than nor less than nor equal to zero, a contradiction has been reached and therefore the gap of prime gaps is aperiodic.


### Theorem: The sequence of all gap levels of primes is aperiodic

In other words, $g_g_g_1, g_g_g_2, g_g_g_4...$ (gap level 3) or any other level gap sequence is aperiodic.

**Proof** A direct extension of the previous theorem: the first gap at any level is odd, but subsequent gaps (on that level) are even, which can only occur if the first gap is never repeated.


### Implications for decidability

As presented [here](https://blbadger.github.io/solvable-periodicity.html), one can show that all decidable predicates may be expressed as periodic systems, or in other words decidability implies periodicity (periodicity with respect to the computation procedure itself, that is).  Taking the contrapositive of this, aperiodicity (in the computation procedure) implies undecidability.  On this page it was found that the sequence of integers of prime gaps are aperiodic, but could some computation procedure for finding prime gaps be itself periodic?

Examining such a procedure, note that it would have to map a countably infinite number of inputs (any of the prime numbers) to a countably infinite number of ouputs (prime gaps, which are unbounded).  The set of all functions possible here is equivalent with the set of all subsets of the natural numbers, $2^ \Bbb N \sim \Bbb R$.  Suppose the mapping function were periodic, meaning that the procedure for finding a prime gap given the prime number's index is identical to the procedure for finding the gap for a different prime number...



If indeed the question of prime gap size is undecidable, this would shed light on the findings from Presburger and Skolem that addition-only or multiplication-only arithmetics are decidable, whereas arithmetic with both addition and multiplication is not (more background [here](https://blbadger.github.io/solvable-periodicity.html)).  The reason is as follows: prime number gaps represent intersections between addition and multiplication, but if these intersections are undecidable (meaning that we cannot compute using finite arithmetic how all gaps should be placed) then arithmetic containing this prime gaps is necessarily undecidable. 

More precisely, undecidability in prime gaps would mean that the question 'Will adding any value to a current number yield another number that cannot be divided (properly)' necessitates a longer and longer computational procedure the larger the number and value are.  Assuming the Church-Turing thesis, decidability is only possible if a finite computational procedure gives the right answer for an infinite number of inputs, here the natural numbers.  Accepting theorems (1) and (2) above together with the idea that there is an equivalence between aperiodicity and undecidability, any arithmetic containing in it the prime gaps will be undecidable.  Note that neither Skolem nor Presburger arithmetic do so: one accepts primes but no gaps (because there is no addition and thus no incrementation), and the other gaps but no primes (no multiplication and therefore no unit of multiplication). 

### Computability 

One can convert decidable statements into computable ones and vice versa by assigning each number one of $\mathfrak t, f$ and then performing a decision procedure.  With this idea accepted, it should be noted that the definition for computability on this page is more strict than the more frequently used definition for this term.  

In that parlance, numbers such as $\pi$ and $\sqrt 2$ are termed computable because there is a procedure that approximates them to any desired accuracy, whereas a number like the Turing halting constant is uncomputable as there is no procedure for approximating it.  It should be noted that even by this broad definition, most real numbers are uncomputable.

On this page, however, computability is defined as the ability to map a countably infinite number of inputs to outputs with a procedure of finite size.  In this sense, a number such as $\pi$ is not strictly computable because the procedure for finding any given decimal digit $d$ increases in size as the distance between the decimel point and $d$ increases. The procedure for mapping each decimal place to its correct digit, in any base, cannot be finite over countably infinite input domain.  This is identical to what is observed for aperiodic dynamical trajectories, where every additional step in time requires an extra computation such that the procedure grows without bound as time increases. 

By this definition, computable numbers or sequences may be determined in arbitrary order, whereas uncomputable ones must proceed in one direction.  To see an example of this, imagine trying to find the 100th digit after the decimal point of the fraction $1/7$ without determining earlier values.  

$$
\frac{1}{7} = 0.142857 \; 142857 \; 1...
$$

We only need to find the identity of the first five digits to determine that the 100th digit is $5$ if we know that $1/7$ is periodic with period 5 in this base.  But trying to accomlish this task for $\pi$ is not possible.

$$
\pi = 3.1415926535...
$$

Any method of approximation by definition start farther and ends closer to the desired value.  As each addition to our decimal list increases the accuracy of approximation, each digit must be determined from left to right in sequence.  If this were not so, we could find the nth digit of $\pi$ or any irrational number with a constant computational procedure.  But to determine (correct) digits out of order is impossible for any approximation technique, because knowledge of one digit necessarily implies knowledge of all the precede it because they are larger.  Only for exact numbers (which are equivalent to computable numbers in this strict definition) may arbitrary digits be learned with constant computation.

Note that this idea also applies to algorithms that do not strictly approximations, such as the Bailey-Borwein-Plouffe [formula](https://en.wikipedia.org/wiki/Bailey%E2%80%93Borwein%E2%80%93Plouffe_formula) for finding desired digits of pi.  Although digits prior to the one of interest are not explicitly calculated, they are implicitly, such that the higher the accuracy desired, the more elements in an infinite series must be added together.  This is true for all spigot algorithms, which find digits (from larger to smaller) in sequence rather than by approximation.  








