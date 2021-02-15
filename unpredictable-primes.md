## Primes are unpredictable

Whilst incrementing through the integers, how long one has to wait until reaching a prime number is often quite a difficult thing to predict.  The question of where the next prime number will be is equivalent to the question of what is the gap between the current prime and the next prime numbers.  As we shall see from two proofs below, it turns out to be impossible to predict how far away the next prime will be without simply computing the primality of each successive integer, either directly or by an interval technique (eg. the sieve of Eratosthenes).  

### Theorem: The sequence of gaps between consecutive prime numbers is not initially periodic

Restated, the sequence between gaps betwen consecutive prime numbers, starting from the first gap $g_1$

$$
g_1 = p_2 - p_1 = 3-2 = 1
$$            

cannot be split into repeated subsequences, such that repetitions of this subsequence cover all prime gaps.  In other words, given a sequence of gaps between consecutive primes $g_1, g_2, g_3 , ... $ there is no way to divide this list into subsequences $ (g_1, g_2, g_3, ... , g_n), (g_{n+1}, g_{n+2}, g_{n+3}, ... g_{2n}), (g_{2n_1}, g_{2n+2}, ... g_{3n}), ... $ such that $g_1 = g_{n+1} = g_{2n+1}, g_2 = g_{n+2} = g_{2n+2}, ... g_n = g_{2n} = g_{3n}$.  

**Proof:** 

Enumerating the prime numbers, we have

$$
p_1, \; p_2, \; p_3, \; p_4, \; p_5, \; p_6, \; p_7, \; p_8 ...\\
2, \; 3, \; 5, \; 7, \; 11, \; 13, \; 17, \; 19 ...
$$

and the gaps between primes where $g_1 = p_2 - p_1$ is enumerated as

$$
g_1, \; g_2, \; g_3, \; g_4, \; g_5, \; g_6, \; g_7 ...\\
1, \; 2, \; 2, \; 4, \; 2, \; 4, \; 2 ...
$$

Now suppose that this sequence of gaps between primes did repeat for multiples of some finite $n > 0$.  Then

$$
(g_1, g_2, g_3, ..., g_n) = (g_{n+1}, g_{n+2}, g_{n+3}, ... , g_{2n}) \\
g_1 = g_{n+1}, g_2 = g_{n+2}, ... g_n = g_{2n}
$$

Adding the sequence of gaps together, we have a sum $\mathscr S_n$ 

$$
g_1 + g_2 + g_3 + \cdots + g_n = \mathscr S_n
$$

which is an integer and thus is either even or odd.  If $\mathscr S_n$ is even, prime $p_{n+1}$ is necessarily even as well because $p_{n+1} = 2 + \mathscr S_n$.  But if $\mathscr S_n$ is odd, then $p_{2n+1} = 2 + 2(\mathscr S_n) = 2(1 + \mathscr S_n)$ and therefore $p_{n+1}$ is even. But this is a contradiction as the only even prime number is $p_1 = 2$ but by definition for any $n > 0$, $p_n > 2$.  Therefore the sequence of gaps between consecutive primes starting from the first prime cannot repeat itself no matter how large $n$ is (remaining finite), or in dynamical terms all subsequences of $g_1, g_2, g_3, ... $ are aperiodic.

### Theorem: Prime gap sequences are not eventually periodic

In symbols, there does not exist finite $n, p$ such that 

$$
\exists n, p : (g_n, g_{n+1}, g_{n+2}, ... , g_{n + p - 1}) \\
= (g_{n+p}, g_{n+p+1}, g_{n+p+2}, ..., g_{n + 2p - 1}) \\
= (g_{n+2p}, g_{n+2p+1}, g_{n+2p+2}, ..., g_{n + 3p - 1}) \\
\; \; \vdots
$$

Note the slightly different enumeration choice as compared with the previous theorem.

**Proof:**

Suppose that there was some finite $n>0$ such that the above statement held.  Then there exists a finite $p$ such that after $p$ gaps, the sequence $g_n, g_{n+1}, ...$ is repeated (with period $p$):

$$
(g_n, g_{n+1}, ... , g_{n+p-1}) = \\
(g_{n+p}, g_{n+p+1}, ... , g_{n+2p-1}) = \\
(g_{n+2p}, g_{n+2p+1}, ... , g_{n+3p-1}) = \\
\; \; \vdots
$$

What happens when this pattern repeats $n$ times?  The prime number indexed at this position is $p_{n+np}$, because there are $n$ primes before $g_n$, and $np$ primes after as in this enumeration $g_n = p_{n+1} - p_n$ means that $g_{n+np} = p_{n + np+1} - p_{n+np}$.   Defining $\mathscr S_n = g_n + g_{n+1} + \cdots + g_{n+p-1}$ as above, the value of $p_{n+np}$ is 

$$
p_{n+np} = p_n + p_n \mathscr S_n \\
p_{n+np} = p_n (1 + \mathscr S_n)
$$

and therefore $p_n \rvert p_{n+np}$, or in words $p_n$ divides $p_{n+np}$.  But then $p_n < p_{n+np}$ for all $n, p>0$, so therefore $p_{n+np}$ is composite, a contradiction as it was ealier stated to be prime. As $n$ and $p$ were chosen arbitrarily, there is no $n$ or $p$ such that the sequence of prime gaps is eventually periodic given that there are infinitely many prime numbers.  

### Theorem: Prime gap sequences are aperiodic

**Proof:** As prime gap sequences are neither initially nor eventually periodic for any finite period $p$, they are aperiodic.

### Implications for decidability

As presented [here](https://blbadger.github.io/solvable-periodicity.html), it is possible to show that aperiodic systems are undecidable: it is impossible to specify a finite computational procedure that can predict the location of any of a countably infinite number of inputs if the mapping is aperiodic.  On the other hand, if the mapping is periodic or eventually periodic then the system is decidable (equivalent to computable if the Church-Turing thesis is accepted) and a countably infinite number of input can be mapped to outputs with a finite computational procedure, or algorithm. A simple example of this is a system which maps all integers to the origin: for any ordering of the inputs, the algorithm is finite (go to 0, to be precise) and interval gaps are of period 1.  

The results on this page shed light on the findings from Presburger and Skolem that addition-only or multiplication-only arithmetics are decidable, whereas arithmetic with both addition and multiplication is not (more background [here](https://blbadger.github.io/solvable-periodicity.html)).  The reason is as follows: prime number gaps represent intersections between addition and multiplication, but if these intersections are undecidable (meaning that we cannot compute using finite arithmetic how all gaps should be placed) then arithmetic containing this prime gaps is necessarily undecidable. 

More precisely, aperiodicity in prime gaps means that the question 'Will adding any value to a current number yield another number that cannot be divided (properly)' necessitates a longer and longer computational procedure the larger the number and value are.  Assuming the Church-Turing thesis, decidability is only possible if a finite computational procedure gives the right answer for an infinite number of inputs, here the natural numbers.  Accepting theorems (1) and (2) above together with the idea that there is an equivalence between aperiodicity and undecidability, any arithmetic containing in it the prime gaps will be undecidable.  Note that neither Skolem nor Presburger arithmetic do so: one accepts primes but no gaps (because there is no addition and thus no incrementation), and the other gaps but no primes (no multiplication and therefore no unit of multiplication). 


### 












