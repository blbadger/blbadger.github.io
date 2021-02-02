## Primes are unpredictable

If one increments through the integers, how long one has to wait until reaching a prime number is often quite difficult to predict.  The question of where the next prime number will be is identical to the question of what the gap between the current prime and the next prime numbers is.  As we shall see from two proofs below, it turns out to be impossible to predict how far away the next prime will be without simply computing the primality of each successive integer, either directly or by an interval technique like the sieve of Eratosthenes.

### Theorem: The sequence of gaps between consecutive prime numbers is not purely periodic

Restated, this theorem is that no sequence of gaps betwen consecutive prime numbers, read from the first gap $g_1$

$$
g_1 = p_2 - p_1 = 3-2 = 1
$$

can never repeat.  In other words, given a sequence of gaps between consecutive primes $g_1, g_2, g_3 , ... $ there is no way to divide this list into subsequences $ (g_1, g_2, g_3, ... , g_n), (g_{n+1}, g_{n+2}, g_{n+3}, ... g_{2n}), ...$ such that $g_1 = g_{n+1}, g_2 = g_{n+2}, ... g_n = g_{2n}$.  

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

In symbols, there does not exist an $n$ such that 

$$
\exists n, p : (g_n, g_{n+1}, g_{n+2}, ... , g_{n + p - 1}) \\
= (g_{n+2p}, g_{n+2p+1}, g_{n+2p+2}, ..., g_{n + 2p - 1}) \\
= (g_{n+3p}, g_{n+3p+1}, g_{n+3p+2}, ..., g_{n + 3p - 1}) \\
\; \; \vdots
$$

Note the slightly different enumeration choice as compared with the previous theorem.

**Proof:**

Suppose that there was some finite $n>0$ such that the above statement held.  THen there exists a $p$ such that after $p$ gaps, the sequence $g_n, g_{n+1}, ...$ is repeated with period p:

$$
(g_n, g_{n+1}, ... , g_{n+p-1}) = \\
(g_{n+p}, g_{n+p+1}, ... , g_{n+2p-1}) = \\
(g_{n+2p}, g_{n+2p+1}, ... , g_{n+3p-1}) = \\
\; \; \vdots
$$

Taking $g_n = p_{n+1} - p_n$ then necessarily $g_{n+np} = p_{n + np} - p_{n+np-1}$.  

Defining $\mathscr S_n = g_n + g_{n+1} + \cdots + g_{n+p-1}$ as above, the identity of $p_{n+np}$ is 

$$
p_{n+np} = p_n + p_n \mathscr S_n \\
p_{n+np} = p_n (1 + \mathscr S_n)
$$

and therefore $p_n \rvert p_{n+np}$, or in words $p_n$ divides $p_{n+np}$.  But then $p_n < p_{n+np}$ for all $n>0$, so therefore $p_{n+np}$ is not prime, a contradiction as it was ealier stated to be prime. As $n$ and $p$ were chosen arbitrarily, there is no $n$ or $p$ such that the sequence of prime gaps is eventually periodic.

### Theorem: Prime gap sequences are aperiodic

**Proof:** As prime gap sequences are neither periodic nor eventually periodic, they are aperiodic.














