## Programs to compute things

### Matrix Determinant

The determinant of a matrix is a number that corresponds to change in volume for a linear transformation encoded by that matrix.  Nonzero determinants are only obtained for square (nxn) matricies, and computing determinants involves a number of patterned multiplication (and addition or subtraction for matricies larger than 2x2) steps.  Computing the determinants of small matricies is not too difficult but it becomes tedious for matricies larger than 3x3.  Imagine computing the determinant for the following matrix:

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
[Finished in 0.4s]```

Which can be checked against software to compute determinants on the web!  Although this program can theoretically compute the determinant of a matrix of any size, it is practically limited to matricies smaller than 11x11 due to time.


### Trailing factorial zeros

The factorial $n!$ is equal to the product of all the natural numbers (not including 0) the same size or smaller than the number in question.  The factorial grows extremely large as $n$ increases, such that the factorial of large numbers are practically uncomputable in a reasonable amount of time!  

But say that we are not interested in the exact value of the factorial, merely how many zeros it has at the end.  









