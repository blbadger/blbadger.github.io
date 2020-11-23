## Roots of polynomial equations

### Newton's method for estimating roots of polynomial equations

Polynomials are equations of the type $ax^b + cx^d + ... = z$ 

$$
x_{n + 1} = x_n - \frac{f(x_n)}{f'(x_n)}
$$



### The simplest unrootable polynomial

At first glance, rooting polynomials seems to be an easy task.  For a degree 1 polynomial $y = ax + b$, setting $y$ to $0$ and solving for x yields $x = -b/a$. For a degree 2 polynomial $y = ax^2 + bx + c$, the closed form expression $x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}$ suffices.  There are more references to the constants (all except $c$ are referenced twice) but there is no indication that we cannot make a closed form expression for larger polynomial roots.  For degree 3 and degree 4 polynomials, this is true: closed form root expressions in terms of $a, b, c ...$ may be found even though the expressions become very long. 

It is somewhat surprising then that for a general polynomial of degree 5 or larger, there is no closed equation (with addition, subtraction, multipliction, nth roots, and division) that allows for the finding of a general root.  This is the Abel-Ruffini theorem.

$$
y = x^5 - x - 1
$$
