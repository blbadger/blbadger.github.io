## Programs to compute things

### Matrix Determinant

The determinant of a matrix (a two-dimensional array) is a number that corresponds to change in volume for a linear transformation encoded by that matrix, and determinants are useful for everything from testing vectors for linear dependance to solving systems of linear equations (which are two tasks that are not really as different as they may seem).  Not all matricies have determinants, but all square matricies do (although the value may be 0).

Determinant can be computed recursively, with the base case being the 2x2 matrix:

$$
\lvert A \rvert = 
\begin{vmatrix}
a & b \\
c & d \\
\end{vmatrix} = ad - bc
$$

The determinant of a 3x3 matrix can be expressed as a linear combination of 2x2 matrix determinants as follows:

$$
\begin{vmatrix}
a & b & c \\
d & e & f \\
g & h & i \\
\end{vmatrix} = 
a
\begin{vmatrix}
e & f \\
h & i \\
\end{vmatrix}
-b
\begin{vmatrix}
d & f \\
g & i \\
\end{vmatrix}
+c
\begin{vmatrix}
d & e \\
g & h \\
\end{vmatrix}
$$

Similarly, determinants of 4x4 matricies can be computed by converting to a combination of 3x3 determinants, which can then be converted to a combination of 2x2 determinants.

Computing the determinants of small matricies using this process is not too difficult, but becomes tedious for matricies larger than 3x3.  The following matrix, for example, would take quite a long time to compute using the recursive method:

```python
matrix = [
[1, 2, 1, 4, 3, -1, 4, 1], 
[1, 2, 3, 2, 9, 1, 10, 2], 
[1, 2, 1, 1, 9, 0, 15, 3], 
[8, 0, 1, 0, 2, 3, 4, -8], 
[2, 3, 4, 0, 1, 2, -1, 3], 
[2, 1, 0, 0, 1, 1, -5, -6], 
[5, -6, 3, 7, -4, 0, 0, 1],
[1, 3, -5, 1, 7, 0, 4, -1]
]
```

Let's try to write a program in Python that can compute matrix determinants for us!  If we review the method of computing the determinant found [here](https://en.wikipedia.org/wiki/Determinant) for a matrix of size 1x1, we see that it is simply the value of this matrix.  For a 2x2 matrix, the computation is also a matter of multiplying and subtracting entries in the matrix.  Things become more difficult for a 3x3 matrix: the computation involves addition and subtraction and multiplication of both matrix entries as well as the determinants of smaller matricies.

Whenever a problem consists of some sort of base case (here the method to compute 1x1 and 2x2 matricies) that is required for other cases, recursion is a good way to proceed.  Our strategy here is as follows: we know how to compute determinants for 1x1 and 2x2 matricies, and the determinants of larger matricies can be computed by reducing down to the determinants of many 2x2 matricies.  

To begin, let's import a useful class `copy` and define our function, including a docstring specifying inputs and outputs.  Now we can add computations for 1x1 and 2x2 matricies such that any time this function is called with a matrix of either size, the determinant is returned.

```python
# standard library
import copy

def determinant(matrix):
    ''' Returns the determinant of a matrix of arbirtary size 
    (note that only nxn matricies have determinants).  Takes
    one argument, a list of lists corresponding to an array
    with numerical values (float or int) for the matrix of interest
    '''
    # base case #1: if the matrix is 1x1
    if len(matrix) == 1: 
        return matrix[0][0]

    # base case #2: if the matrix is 2x2
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
 ```

For larger matricies, we first need to construct slightly smaller matricies (without the row and column of each of the entries in the top row) which will then be further reduced if they are larger than 2x2 or else their determinants will be found.  First we assign the variable `result` to 0, which become our determinant value, and then make a loop that will iterate through the matrix width.  Next we use some list comprehension to make a new matrix with the first row removed, and copy this (to prevent problems with variable assignment in the recursion) and remove the column corresonding to the particular place in the first row designated by `i`.

```python
    else:
        result = 0
        for i in range(len(matrix)):
            new_matrix = [matrix[j] for j in range(len(matrix)) if j != 0]
            new_matrix2 = copy.deepcopy(new_matrix)
            for row in new_matrix2:
                del row[i]
```

Now comes the recursion: using the rule that every other determinant is added (and the rest are subtracted) as one iterates through the top row, the function `determinant()` is called on the smaller matrix made above.  If this matrix is larger than 2x2, the process continues until the determinant can be directly computed, whereupon it is multiplied to the appropriate top row value and subtracted or added to the variable `result`.  Outside the `else` clause, we return the result and are all done!

```python
            if i%2 == 0:
                result += matrix[0][i] * determinant(new_matrix2)
            else:
                result -= matrix[0][i] * determinant(new_matrix2)

    return result
```

Let's test the program out on our large matrix!  By printing out the results using `print (determinant(matrix))`, we get
```python
355329
[Finished in 0.4s]
```

Which can be checked against the built-in matrix determinant calculator found in numpy:

```python
import numpy

m = numpy.matrix(matrix)
print (numpy.linalg.det(m))

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
355328.9999999997
[Finished in 0.2s]
```

Although this program can theoretically compute the determinant of a matrix of any size, it is practically limited to matricies smaller than 11x11 due to time.

Can we make this program faster?  Thinking carefully about what the recursion is doing, we can see that the same computations will be performed over and over again.  This is a perfect opportunity to employ some heroic dynamic programming in order to save the computations we did previously in memory and simply refer to the answer we got last time we performed the computations.

To begin, we add a dictionary that will store our computed results.
```python
# standard library
import copy
matrix_dictionary = {}
```
we call the a function with the dictionary as an argument.  The base cases are the same (and omitted here for brevity) but then insteady of simply calling the smaller matrix determinant, we instead only do so if a tuple version of the matrix does not exist in our dictionary (it must be a tuple because lists are mutable and python dictionaries can only hash non-mutable datatypes like strings or tuples).  The determinant of this smaller matrix is saved in the dictionary such that whenever we want to know what the determinant is of this matrix is in the future, we can just look it up instead of recalculating it. 

```python
def determinant(matrix, matrix_dictionary):
    ... 
    ...
    # iterate through the values of the matrix, removing the row and 
    # col of each value contributing to the determinant. Find the 
    # value recursively only if it has not already been found and added to 
    # matrix_dictionary.  If it has already been computed, it is added to 
    # the dictionary.
    else:
        result = 0
        for i in range(len(matrix)):
            new_matrix = [matrix[j] for j in range(len(matrix)) if j != 0]
            new_matrix2 = copy.deepcopy(new_matrix)
            
            for row in new_matrix2:
                del row[i]

            new_tuple = tuple(tuple(i) for i in new_matrix2)
            if i % 2 == 0:
                if new_tuple not in matrix_dictionary:
                    new_matrix_det = determinant(new_matrix2, matrix_dictionary)
                    result += matrix[0][i] * new_matrix_det
                    matrix_dictionary[new_tuple] = new_matrix_det
                else:
                    result += matrix[0][i] * matrix_dictionary[new_tuple]
            else:
                if new_tuple not in matrix_dictionary:
                    new_matrix_det = determinant(new_matrix2, matrix_dictionary)
                    matrix_dictionary[new_tuple] = new_matrix_det
                    result -= matrix[0][i] * new_matrix_det
                else:
                    result -= matrix[0][i] * matrix_dictionary[new_tuple]

    return result
```

When we compare the running time of the pure recursive version with the 10x10 matrix:
```python
matrix = [
[1, 2, 1, 4, 3, -1, 4, 1, 5, 10], 
[1, 2, 3, 2, 9, 1, 10, 2, 9, -4], 
[1, 2, 1, 1, 9, 0, 15, 3, 5, -3], 
[8, 0, 1, 0, 2, 3, 4, -8, 3, -8], 
[2, 3, 4, 0, 1, 2, -1, 3, 1, -3], 
[2, 1, 0, 0, 1, 1, -5, -6, 1, 1], 
[5, -6, 3, 7, -4, 0, 0, 1, 2, 6],
[1, 3, -5, 1, 7, 0, 4, -1, 3, 0], 
[3, 4, 2, 7, 2, 1, 5, -9, 3, 13], 
[6, 7, -4, 1, 0, 1, 9, -1, 3, 1],
]
```
which yields

```bash
(base) bbadger@bbadger:~/Desktop$ time python matrix_determinant.py
18639282

real	0m31.581s
user	0m31.565s
sys	0m0.012s

```
to the running time of the top-down dynamically programmed version (using memoized recursion),
```bash
(base) bbadger@bbadger:~/Desktop$ time python matrix_determinant_memoized.py
18639282

real	0m0.169s
user	0m0.157s
sys	0m0.012s
```

we get the same answer, but the memoized version of the program is much faster! This one calculates matricies of under 19x19 in a reasonable amount of time, a substantial improvement over the standard recursive program.

The memoized version is a little faster than the numpy `numpy.linalg.det()` for this matrix 

```bash
(base) bbadger@bbadger:~$ time python ~/Desktop/matrix_determinant_memoized.py
18639282.00000001

real	0m0.195s
user	0m0.323s
sys	0m0.221s
```

For larger matricies, `numpy.linalg.det()` is faster than our memoized recursive solution.  How is this possible?  It turns out that while the recursive approach to matrix determinant finding is perfectly good, this is not the fastest approach: instead, there is a properties of the determinant that allow for an approach that does not use recursion at all, and is O(m+n) time complexity.  This property is that scalar multiple of one column to another does not change the determinant's value.  Therefore, determinants can be computed very rapidly by simply eliminating columns (transforming values to zeros in that column) before proceeding with the above recursive algorithm.

### Trailing factorial zeros

The factorial $n!$ is equal to the product of all the natural numbers (not including 0) the same size or smaller than the number in question.  The factorial grows extremely large as $n$ increases, such that the factorial of large numbers are practically uncomputable in a reasonable amount of time!  

But say that we are not interested in the exact value of the factorial, instead the quetions is how many zeros the number has at the end.  To make this as general as possible, let's extend this question to how many zeros the factorial of a number `n` has at the end, in a given base `base`. For example, 5 factorial in base ten is 120, which has one trailing zero.

$$ 
5!_{10} = 12\mathbf{0} \to 1 
$$

whereas 20 facotorial in base 3 has 8 trailing zeros

$$
20!_{3} = 1210121221100100101120122022221\mathbf{00000000} \to 8 
$$

How is one to find the number of trailing zeros of a factorial without calculating the entire number?  Every natural number is composed of prime numbers, so looking at primes might be a good idea.  Let's start in base 10, which is the most familiar for many of us.  Every number in base 10 may be represented by a string of digits 0-9.  Taking a factorial of a number is the same as multiplying all smaller natural numbers together, so we are interested in how to get 0s at the end of a number as the result of multiplication. 

So multiplication is the key transformation we are applying to a pair of numbers at a time, but which digits contribute to trailing 0s?  The last digit of each number is important here, because if the last digits do not multiply together to make 0 then there are no trailing 0s!  Now we can ask: which primes in range 0-9 (ie which digits) multiply to make a 0?  The answer is 2 and 5, and although 0 is not a prime it can be included in this list because it will also yield a 0. 

Now the question is slightly different: how do we find the number of 5, 2, and 0s that will be multiplied together in the last non-zero digit of the factors of $n!$?  Fortunately for us, there exists a formula for determining the largest power of any given prime number of the factorial of a number: [Legendre's formula](https://en.wikipedia.org/wiki/Legendre%27s_formula).  This formula is wonderful because it allows us to bypass calculating the trailing zeros factor by factor and simply focus on the single factorial number itself.  Legendres formula is as follows: for a prime number $p$ and factorial $n!$, $v_p(n!)$ is the exponent of the largest power of prime $p$ that divides $n$ without remainder and is computed as follows:

$$
v_p(n!) = \sum_{i = 1}^\infty \lfloor\frac{n}{p^i}\rfloor
$$

For the general case in any base $p$, the formula becomes

$$
v_p(n!) = \frac{n - s_p(n)}{p-1}
$$

Where $s_p(n)$ is the sum of the digits of the base-$p$ version of $n$.

Now we are ready to tackle the problem of trailing zeros!  First we will need to define a function (including a doc string) and in this function, make a helper method to determine if a number is prime. The library `math` is also imported, as it contains many useful mathematical functions.  The function to test whether a number is prime or not is fairly straightforward: going from 2 up to the square root of the number in question, if this number is divisible by any of these smaller ones then it is not prime. We can stop at the square root of the number in question because any larger number that is a factor of our original must also have a factor smaller than the square root as well, and this was already checked.

```python
# Import standard library
import math

def zeros(base, n):
    '''Returns the number of 0s trailing the factorial of a number (n) supplied in any given base (argument base)
    using Legendre's method.  This allows for the number of trailing zeros to be computed for inputs whose factorial
    is not computable in any reasonable amount of time.
    '''

    def is_prime(number):
        '''Determines if the argument number is prime.
        Outputs True if prime, else False.
        '''
        if number == 2:
            return True
        for i in range(2, int(number**0.5) + 1):
            if number % i == 0:
                return False
        return True
```

Now let's consider the effect of different bases on the number of trailing zeros.  Say we are in base 10.  Digits 5 and 2 multiply to make 10, and so add a zero. But in base 7, no smaller primes compose 7 because it is prime and so the only digit that can make more zeros in this base is 7 and 1.  We assign an empty list to `ls`, which will store the primes that compose our base and the number of times they are multiplited to make the base. 

If the base is not prime, we find its composition.  Ranging from 2 to half the base, we can use our `is_prime()` function to determine if a digit smaller than `base` is prime.  If it divides `base` evenly (with no remainder), then we incrementing `j` until it no longer does.  We then assign the value of `j` (minus one to account for the initial value of 1) to the variable `exponent`, and add the prime `i` with the exponent associated with this prime we just found by dividing `base` repeatedly.  At the end of the `for` loop, we have a `ls` with all primes that compose the `base` together with the exponents that multiply these primes.  Adding each prime `i` to the power of its `exponent` should return our `base`.  If the base is prime, we add only the `base` with `1` to the list.

```python
    exponent = 1
    ls = []
    if not is_prime(base):
        for i in range(2, base//2 + 1):
            if is_prime(i) and base%i == 0:
                j = 1
                while base % (i ** j) == 0:
                    j += 1
                exponent = j - 1
                ls.append([i, exponent])
    else:
        ls.append([base, 1])
```

So now we have the composition of our base, in the form of a list `ls`.  

Now it is time for another helper function. We want to find the sum of the digits of $n$ for the general Legendre's formula, so the function `sum_digits(base, n)` is made to compute the sum of the digits of $n$ in the base $base$ provided. 

```python
  def sum_digits(base, n):
        '''A helper function that adds the digits 
        of the argument n in the base provided.
        '''
        number_ls = []

        while n >= base:
            remainder = n % base
            number_ls.append(remainder)
            n -= remainder
            n = int(n // base)
        number_ls.append(n)
        return sum(number_ls)
```

Now we can put these pieces together and apply Legendre's formula!  First we can assign a variable `power_ls` to an empty list.  Then loop over `ls`, storing the exponent of each prime in `base` (`pair[0]`) for this prime in $n1$ as determined by the general Legendre's formula.  This number is assigned to the variable `power`, which is then divided by the exponent of that prime in the composition of `base` (`pair[1]`).  We use floor division here because the interest is in the maximum number of times the prime composition `pair[1]` can fit in `power`.  This result is added to `power_ls`, and the loop continues for all values in `ls`.  The minimum of `power_ls` is returned because each zero in the factorial requires at least one of each prime factor, so the minimum power of these factors gives us the number of trailing zeros in our base of interest.  

```python
    # Legendre's method for computing trailing zeros 
    power_ls = []
    for pair in ls:
        power = int((n-sum_digits(pair[0], n))/(pair[0] - 1))
        power_ls.append(power // pair[1])

    if not power_ls: return 0
    
    return min(power_ls)
```

And now we are done! Let's check the work.

```python
# example inputs
base = 3
n = 20

# example function call
print (zeros(base, n))

~~~~~~~
8
[Finished in 0.1s]
```

It checks out for $20!$ in base $3$.  More tests can be done (unit testing is particularly useful here) to convince us that this function does what it says it does.  Does it run in reasonable time for large values of n?  It does!

```python
# example inputs
base = 7
n = 1349182374091283740932184

# example function call
print (zeros(base, n))

~~~~~~~~~~~~
224863729015213959151616
[Finished in 0.0s]
```


