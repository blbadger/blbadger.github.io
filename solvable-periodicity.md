### Decidability and periodicity

[Elsewhere](https://blbadger.github.io/aperiodic-irrationals.html) we have seen that the set of all periodic dynamical systems (albeit systems as defined in a specific manner) is equinumerous with the set of solvable problems.  The set of all aperiodic systems was found to be more numerous, instead equivalent to the size of the set of all unsolvable problems.  This page will investigate a direct equivalence between the the members of the sets of solvable problems and periodic maps to arrive at a stronger conclusion: equality (with substitution) between these sets, rather than equivalence.

'Unsolvable' and 'Undecidable' are here defined in terms of mathematical logic: decidability is the process of determining the validity of a $\mathfrak t, \mathfrak f$ (true or false) statement.  Validity means that the statement is always true, regardless of the inputs to the statement.  Solvable means that there is some computation procedure that applies to the given function (or problem) that is finite in length.  

Decidability and solvability are equivalent in number theory. The solvable problem $x = 2^2$ is identical the the decision problem 'Is 4 the square of 2?'.  For this page, solvable implies a 'for-what?' statement and decidable implies a 'is it true?' statement.  The law of the excluded middle is accepted, meaning that a proposed statement must be true or false but not neither. In symbols,

$$
\exists x P(x) \lor \lnot \exists x P(x)
$$

### Decadiability and the Church-Turing thesis

Church and Kleene postulated that all effectively computable functions are $\lambda$-definable, and later shown that these functions are equivalent to the general recursive functions Godel defined to be computable.  

A particularly poignent implication of the Church-Turing thesis is that most functions are uncomputable by any means.  A proof for this is as follows, from Kleene's *Mathematical Logic*:

Theorem (1): Most functions are not effectively computable

Proof: There are an uncountably infinite number of 1-place number theoretic functions on a countably infinite number of inputs.  Suppose that we could enumerate all such functions in a list $f_0(a), f_1(a), f_2(a), ...$.  Each function has the domain of the natural numbers $0, 1, 2, 3, ...$ so we can list them as follows:

$$
f_0(a): f_0(0), \; f_0(1), \; f_0(2), ... \\
f_1(a): f_1(0), \; f_1(1), \; f_1(2), ... \\
f_2(a): f_2(0), \; f_2(1), \; f_2(2), ... \\
...
$$

Now define a function $f_n(a) = f_a(a) + 1$.  This function is an application of Cantor's diagonal method: it differs from $f_0(a)$ because $f_n(0) = f_0(0) + 1$ and no number is its own successor.  Similarly, it differs from the other functions listed because $f_n(1) = f_1(1) + 1$ and $f_n(2) = f_2(2) + 1$.  No matter how many functions we enumerate, $f_n(a)$ will be different from them all.  But as $f_n(a)$ is a one-place number theoretic function, we have a contradiction and conclude that there is no enumeration of all 1-place number theoretic functions. 

Now as there are only countably infinite Turing machines (see below), because each machine is defined by a finite list of instructions and these lists can be enumerated one by one.  Therefore there are many fewer Turing machines than possible number-theoretic functions, and as each Turing machine describes one number-theoretic function then necessarily most functions are uncomputable on Turing machines (or equivalently in general recursive functions or by $\lambda$ calculus).  If we accept the Church-Turing thesis, this means that most functions are not computable by any means (ie they are not effectively computable).  'Most' here is an understatement, as actually only an infinitely small subset of all number theoretic functions are computable.

### Computability with Turing machines

To gain more appreciation for what was presented in the previous section, a closer look at Turing machines is helpful. A [Turing machine](https://en.wikipedia.org/wiki/Turing_machine) is an abstract system whose rules were designed by Turing to model computation.  A tape of infinite length has a sequence (of 0s and 1s for two state Turing machines) with symbols printed on it is fed into this machine and the output is the tape but with the symbols re-written.  Each step of a Turing machine procedure involves performing an action (or not), moving to the left or right (or staying stationary), and then changing state (or staying in the same state).  Turing machines are described by finite tables of instructions that completely specify what the machine should do at any given point.  For example, the following 1-state, 2-symbol machine if starting on a '1', changes the symbol to a '0', moves right, and halts:

$$
0 \; \; \; \; \; \;  1  \\
1\; EC0 \; \; ER1
$$

where 'P' is print '1', 'E' means erase (print 0), 'R' means move right, 'L' means move left, 'C' means stay in the same place, and the final number specifies the state of the machine ('0' or 'H' means halt).  The variable in question is the column head, and the row (here only '1') denotes state.

Now simply changing a symbol does not seem very useful, if one accepts the Church-Turing thesis then Turing machines are capable of performing any computational procedure that we can define. A more complicated one, found in Kleene 1967, takes any integer and adds one.  The instruction table is as follows:

$$
\; \; 0 \; \; \; \; \; 1 \; \\
1 \; C0 \; \; \; R2 \\
2 \; R3 \; \; \; R9 \\
3 \; PL4 \; \; \; R3 \\
4 \; L5 \; \; \; L4 \\
5 \; L5 \; \; \; L6 \\
6 \; R2 \; \; \; R7 \\
7 \; R8 \; \; \; ER7 \\
8 \; R8 \; \; \; R3 \\
9 \; PR9 \; \;  L10 \\
10\;C0 \; \; ER11 \\
11\;PC0 \; \; R11
$$

which may emulated as a Turing machine using c++ as follows:

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
using namespace std;

vector<int> turing_incrementor(vector<vector<string>> program, int state, int position, vector<int> input, int& counter){
	
	// initialize the result tape as a copy of the input tape
	vector<int> result;
	for (auto u: input) result.push_back(u);
	
	while (true) {
		int square = result[position];
		string step;
		
		if (square == 1) {
			step = program[state-1][1];
		}
		else step = program[state-1][0];
		
		int index = 0;
		if (step[0] == 'E' or step[0] == 'P'){
			index = 1;
			if (step[0] == 'P') {
				result[position] = 1;
			}
			else result[position] = 0;
		}

		if (step[index] == 'R') position += 1;
		else if (step[index] == 'L') position -= 1;
		
		state = 0;
		string num_string = "";

		index++;
		while (index < step.size()) {
			num_string.push_back(step[index]);
			index++;
		}
		state = stoi(num_string);
		counter++;
		if (step[step.size()-2] == 'C' and step[step.size()-1] == '0') {
			return result;
		}
	}

}
	
int main(){
	//initialize program for incrementing integers
	vector<vector<string>> program { {"C0", "R2"}, {"R3", "R9"}, 
		{"PL4", "R3"}, {"L5", "L4"}, {"L5", "L6"}, {"R2", "R7"}, 
		{"R8", "ER7"}, {"R8", "R3"}, {"PR9", "L10"}, {"C0", "ER11"}, 
		{"PC0", "R11"} };
	
	//initialize input tape
	vector<int> input;
	
	//specifiy input number 
	int initial_number = 1000;
	for (int i=0; i<2*initial_number + 10; i++) input.push_back(0);
	for (int j=3; j<initial_number+3; j++) input[j] = 1;
	cout << "Input:   ";
	for (auto u:input) cout << u;
	//state initialization
	int state = 1;
	
	//counter initialization
	int counter = 0;
	
	int position = 3 + initial_number-1;
	vector<int> result;
	
	result = turing_incrementor(program, state, position, input, counter);
	cout << counter;
	cout << endl;
	cout << "Output:  ";
	for (auto u:result) cout << u;
	return 0;
}
```

The input is a series of '1's corresponding to an integer and the output is this followed by '0' and then the incremented integer.  For example, and input of  `00011000000`, corresponding to the integer 1 (as `000100` is 0) becomes `00011011100`, which is 1 followed by 2.  

This program is capable of incrementing any finite number, and so clearly the function of incrementing 1 is computable.  But in the last section, it was claimed that there are number theoretic functions that are uncomputable by Turing machines, or by any other effective method (Church's lambda calculus etc.) if the Church-Turing thesis is accepted.  What are these functions that are not computable, and are any of these uncomputable functions important?  

Let's introduce a useful predicate function: $T(i, a, x)$ signifies a Turing machine, with index integer $i$, input argument integer $a$, that returns $\mathfrak t$ (true) if the computation halts at step $x$ but otherwise $\mathfrak f$.  The input argument $a$ is any integer, which when encoded could be `000100` signifying $a=0$ etc.

The Turing machine index $i$ is defined as the instructions for the Turing machine, encoded as an integer (which is usually very large).  For example, the table of instructions for the Turing machine above that increments integers can be represented as the integer

$$
21136737094333657700134424238322063624969284419 \\
08675647661770484612266829110943192693004390163 \\
75863429484413035036069094311865821860452564289 \\
359749323778362522231196047032053805361
$$

Is $T(i, a, x)$ a decidable predicate function, or in other words is it computable?  It is: given any Turing machine table (in integer form) $i$, and any input $a$, we can simply run the machine until step $x$ to see if the machine stops at that step or not, and return $\mathfrak t$ or $ \mathfrak f$.  

To compare across an infinite number of Turing machines without having to deal with an infinite number of different inputs, we can simply set the input argument $a$ to the machine's index $i$, or $i = a$, which means that the input to the program is the program itself. Now $T(a, a, x)$ is the true or false predicate that the Turing machine with input $a$ and index $a$ halts at step $x$, and is computable for the same reason $T(i, a, x)$ is.  But what about the question of whether there is any number of steps $x$ such that this machine halts, $\exists x \; T(a, a, x)$: is this computable too?  We can no longer simply run the Turing machine to find an answer because we are interested in all possible $a$ values.  

Happily, however, one can list all possibilities of $T(a, a, x)$ as follows:

$$
T(1, 1, 1), \; T(1, 1, 2), \; T(1, 1, 3)  \cdots \\
T(2, 2, 1), \; T(2, 2, 2), \; T(2, 2, 3)  \cdots \\
T(3, 3, 1), \; T(3, 3, 2), \; T(3, 3, 3)  \cdots \\
\vdots
$$

because any Turing machine's instructions may be represented as an integer, and the same for any input to the same machine, it should be apparent that this table includes all possible Turing machine instructions and all possible inputs. Finding whether or not the machine halts at step $x$ is simply a matter of looking up the value on the table.

To restate the question at hand, can we compute the function $\exists x \; T(a, a, x)$, for any input $a$ and index $a$?  If we could do this, then we could create a Turing machine that simply adds one step (move right and halt, perhaps) to each machine $T(a, a, a)$ for all $a$.  Therefore if $T(a, a, a) = \mathfrak t$ then $T(a, a, a+1) = \mathfrak f$ and if $T(a, a, a) = \mathfrak f$ then $T(a, a, a+1) = \mathfrak t$ because there is only one integer number of steps $x$ at which the Turing machine halts.  

Clearly this machine returns a different $\mathfrak t, \mathfrak f$ result than each machine listed above for all values of $a$ where $T(a, a, a)$: if for example a certain encoding gave

$$
T(1, 1, x) = \mathfrak t, \; \mathfrak f, \; \mathfrak f \cdots \\
T(2, 2, x) = \mathfrak f, \; \mathfrak t, \; \mathfrak f \cdots \\
T(3, 3, x) = \mathfrak f, \; \mathfrak t, \; \mathfrak f \cdots \\
$$

then this new Turing machine would be

$$
T'(1, 1, x) = \mathfrak t, \; \mathfrak f, \; \mathfrak f \cdots \\
T'(2, 2, x) = \mathfrak f, \; \mathfrak f, \; \mathfrak t \cdots \\
T'(3, 3, x) = \mathfrak f, \; \mathfrak f, \; \mathfrak t \cdots \\
$$

which is a different table by definition.  Different Turing machine outputs (for the same inputs) can only occur for different machines, so we know that $T'$ differs from all Turing machines listed above whenever $a=i=x$, or at the $\star$ machines in the table:

$$
T\star(1, 1, 1), \; T(1, 1, 2), \; T(1, 1, 3)  \cdots \\
T(2, 2, 1), \; T\star(2, 2, 2), \; T(2, 2, 3)  \cdots \\
T(3, 3, 1), \; T(3, 3, 2), \; T\star(3, 3, 3)  \cdots \\
\vdots
$$

But this means that the new Turing machine is not present on this list, because it differs from each entry for at least one input.  The list by definition included all possible Turing machines, so a contradiction has been reached and the premise that $\exists x \; T(a, a, x)$ is computable must be false.


### Busy Beaver Turing machines

Code for this section is found [here](https://github.com/blbadger/turing-machines).

In other words, whether or not an arbitrary Turing machine (with an input equal to its instructions) is going to ever halt is not a decidable predicate, even though it is clear for any finite step $x$ whether the machine halts there.  This does not mean that if we are given a value of $a=i$, we cannot determine if the machine halts in some cases.  The first machine presented, which simply changes the initial tape entry and moves to the right and halts, will always halt.  Likewise a machine that only moves to the left without any other instructions clearly will not halt.  But $\exists x \; T(a, a, x)$ contains $a$ as a free variable, meaning that it can take any value.  The surprising conclusion then is that there is no way to determine whether a machine halts given an arbitrary value of $a$, although for certain $a$ the answer is clear.

To better understand just how unclear it is whether an arbitrary program will halt, Rad&oacute; described the busy beaver challenge: to make the longest number of steps to a Turing machine of a given size that does not obviously run forever.  For two state Turing machines (with entries of 1 or 0), the maximum number of steps an eventually-stopping program takes is 6.  This number must be found by trial and error, because as presented in the last section whether or not a program halts for all possible inputs is not computable.  In the language used above, this program is

$$
0 \; \; \; \; \; 1 \; \\
1 \; PR2 \; \; PL2 \\
2 \; PL1 \; \; PRH \\
$$

which given an input of $...0000000000...$ returns $...00001111000...$ (depending exactly where the starting index is).  For three states, the maximum number of steps of an eventually-stopping machine is 21, which also does not seem so unmanageable.  But for five possible states and two symbols, the machine table

$$
\; 0 \; \; \; \; \; 1 \;  \\
1\; PR2\; \; PL3 \\
2\; PR3\; \; PR2 \\
3\; PR4\; \; EL5 \\
4\; PL1\; \; PL4 \\
5\; PRH\; \; EL1 \\
$$

takes 47176870 steps to complete.  As found in the very useful repository [here](http://www.logique.jussieu.fr/~michel/ha.html#tm52), the maximum number of steps for a six-state, two symbol Turing machine is greater than $7.4 × 10^36534$.  The true value is unknown, because this machine (and a number of other similar ones) are still running, with no indication of looping or stopping.  And this is for only 6 possible states and 2 possible symbols: imagine how many steps it would require to find whether all possible 11-state, 2-symbol machines halt!  An yet this number of states was used for something as simple as integer incrementation.  Clearly then, without some kind of external cue as to whether a machine halts or not the answer is unclear.


### Unsolvability and aperiodicity

Solvable problems (ie computable functions) are those for which, by the Church-Turing thesis, there exists an algorithm of finite size for a Turing machine that yields a decision for any of a countably infinite number of decision inputs.  

Take the mapping of Turing machine (or $\lambda$ -recursive function, as in Church's original thesis) inputs to the outputs as a dynamical system, where each element of the set of countably infinite inputs $\{ i_0, i_1, i_2 ... \}$ is mapped to its output $ \{ O_0, O_1, O_2 ... \}$ as 

$$
\{i_0 \to O_0, \; i_1 \to O_1, \; i_2 \to O_2 ...\}
$$

To be solvable, an algorithm of finite length must be able to map all of the countably infinite inputs.  After a certain number of $n$ inputs, the decision procedure for $i_0$ must be identical to the decision procedure $\mathscr P$ for $i_{0+n}$ or else the algorithm would be of infinite size, meaning that it would be unusable as it would never halt when applied to a Turing machine.  Thus $\mathscr P$ repeats after a finite interval $n$ between inputs.  

As the inputs are countable, they can be arranged in sequence.  Taking the sequence of decision procedures $\mathscr D$ for all inputs, the function mapping an input $\{ i_0, i_1, i_2 ... \}$ to its decision procedure $ \{ \mathscr D_0, \mathscr D_1, \mathscr D_2 \}$ defined as $f(i)$ can be viewed as a discrete dynamical system.  

Theorem (2): For all decidable problems, $f(i)$ is periodic for some ordering of $i_0, i_1, i_2 ...$.

Proof: Suppose that $f(i)$ is computable but is not periodic for any ordering of $i_1, i_2, i_3 ...$.  Then

$$
\forall n, k : n \neq k, \; \mathscr D_n \neq \mathscr D_k
$$

Therefore there are a countably infinite number of elements $\mathscr D_0, \mathscr D_1, \mathscr D_2 ...$ meaning that $f(i)$ is countably infinite.  Thus $f(i)$ would be infinitely long, but this contradicts the definition of computability.  Therefore $f(i)$ is periodic for some ordering of $i_1, i_2, i_3 ...$ (which is equivalent for stating that $f(i)$ is periodic for some ordering of its elements $\mathscr D_0, \mathscr D_1, \mathscr D_2 ...$ ).  

Thus $f(i)$ gives us the function mapping each computable function (or decidable predicate or solvable problem) to one discrete dynamical equation.  

### What does it mean that most problems are unsolvable, or that most $\mathfrak t, \mathfrak f$ predicates are undecidable?

The appearance of undecidable predicates and unsolvable problems (and unprovable but true statements) in arithemtic with the natural numbers is disconcerting enough when one considers how intuitively consistent the axioms of this system are as compared to, say, those of analysis.

On the other hand, consider this: in the real world, many problems appear to be so difficult to find a solution to as to be, for all intents and purposes, impossible.  The study of mathematics attempts to understand the behavior of often very abstract ideas, but it is rooted in a study of the natural world: of shapes made by objects here on earth, paths that stars take in the sky, and the like.  Great utility of mathematics in describing the natural world remains to this day.  

Thus mathematics would be expected to reflect something about the natural world.  Now if all mathematical problems were solvable, it would not seem that this study was an accurate description of the world around us.  If a system with axioms as simple as arithmetic could have unsolvable problems, there is hope that even something as historically removed from application as number theory is able to capture something of reality.

### Addition-only or multiplication-only number theories are decidable

Skolem defined an arithemtic with multiplication but not addition, and showed that this is decidable and complete.  Similarly, Presburger defined an arithemtic with addition but not multiplication and found this was decidable and complete.

These results are surprising because addition composes multiplication.  For any arbitrary multiplication function $m(x)$ and addition function $a(x)$, we can compose the multiplication function of repeated additions as follows:

$$
m(x) = nx \implies m(x) = a \circ a \circ a \cdots \circ a = a^n(x)\\
$$

Similarly, division $d(x)$ on the integers can be composed of repeated applications of addition $a(x)$ as follows:

$$
a \circ a \circ a = a(a(a(n))) \\
d(x) = \frac{x}{n} \implies d(x) = card \; \{a \circ a \circ a \cdots \circ a \} :  \\
a \circ a \circ a \cdots \circ a = x \\
a^{d(x)}(n) = x
$$

In other words, division of one number by a second is equivalent to the number of times addition must be composed on the second number to equal the first, or the cardinality of the set of addition compositions required to transform the second number into the first.

As subtraction is defined as the inverse of addition, all four arithemetical operations may be performed on terms of only addition.  How then is addition-only or multiplication-only number theory decidable, but if both operations are allowed then the theory is undecidable?

A direct answer to this question is difficult, but an interesting analogy is available with dynamical systems.  To restate, number theory with only addition is decidable, as is number theory with only multiplication but with both addition and multiplication, the theory is undecidable.  In dynamics, as seen elsewhere on [this page](https://blbadger.github.io/), transformations involving only addition are linear and solvable, and transformations of only multiplication (if bounded) are nonlinear but also solvable, as they simply expand or compress the function pre-image.  But transformations with both addition and multiplication may be aperiodic and unsolvable.

### Examples of undecidably-undecidable number theoretic statements

Kleene found levels of undecidabity, implying some statements are more undecidable than others.  Some problems, like whether or not a Turing machine will halt given an arbitrary input, are demonstrably undecidable.  But other problems may or may not be undecidable, and these would be expected to occupy a higher level of undecidability with respect to the halting problem.  Some such probably-undecidable statement examples are put forth here.  

As both Presburger and Skolem arithmetics are decidable (and complete) but arithmetic with both addition and multiplication is not, one would expect to find some undecidable number-theoretic statements existing at the intersection of multiplication and addition.  If we consider prime numbers to the the 'atoms' of multiplication, this is indeed the case: there are a number of as-yet-undecidable statements relating primes to addition. 

First and most famous is Goldbach's conjecture, that all integers greater than two are the sum of two primes,

$$
\forall n>2, \; n \in \Bbb N \; \exists a, b \in \{primes\} : n = a + b
$$

There is also the twin primes conjecture, that there is an infinite number of pairs of primes that are two apart from each other,

$$
\lvert \{p, p+2\} \rvert > n \; \forall n\in \Bbb N : p, p+2 \in \{ primes \}
$$

and Legendre's conjecture, that there exists a prime number between the square of any integer and the square of its successor,

$$
\forall n \in \Bbb N, \; \exists p \in \{ primes \} : n^2 < p < (n+1)^2
$$

All are as yet unproved and yet seem true, at least they are true for all inputs observed thus far.


### Computability and the axiom of choice in Banach-Tarski

Cantor's set theory has become foundational for many areas of mathematics, but in its original form (naive set theory) the theory was observed to harbor contradictions that were unsatisfactory to mathematicians at the turn of the 20th century.  Axiomatized set theory was an attempt to prevent many such contradictions, and did this by replacing intuitive notions of 'collections' with rules for what sets could be composed of.  

One particular axiom of interest is the axiom of choice, which states that the Cartesian product of two aritrary non-empty sets is non-empty. Equivalently, there exists some function to find any value of choice from even an infinitely large set.  This axiom was controversial when first introduced but is not widely accepted. The axiom is similar to the law of the excluded middle detailed at the top of this page, in that one seems to be able to search through an infinite set with a function to find an element just as with the law of the excluded middle, one of $\mathfrak t, \mathfrak f$ is chosen even if the function domain is infinte (meaning that we cannot practically test every element).

The axiom of choice gives rise to an interesting, and somewhat counterintuitive idea: the Banach-Tarski theorem, which states that an arbitrary set-theoretic sphere of infinite points can be decomposed and recomposed into two spheres of the same 'size'.  The idea that one ball can be split up and the re-formed into two of the sme size is also called the pea and the sun paradox, because with repeated applications of Banach-Tarski a small pea could be multiplied to the size of a large object like the sun.

A reconciliation with this paradox may be had if one observes that the function required to perform the dissassembly and reassembly of spheres in Banach-Tarski is uncomputable.  This follows from how the deconstruction and reconstruction is performed: it is an infinite recursive (tree) algorithm, and therefore not computable if we accept the Church-Turing thesis (or if we are reasonable people in general). 

If the Church-Turing thesis is correct and our somewhat undefined notion of computability is captured by Turing machines (or by general recursive functions), then the Banach-Tarski theorem is paradoxical in part because it relies on functions that we cannot compute.  

### Aside: computability and the natural world

Does it matter that most functions are uncomputable (accepting the Church-Turing thesis)?  Sometimes it is argued that no, just as the rationals are dense in the reals so computable functions can approximate any function. This argument relies on assumptions of approximations that do not hold for nonlinear functions and is therefore unhelpful as most functions are nonlinear. 

Another argument is that most of these uncomputable functions would mostly resemble mappings to (white or other) noise, like the static on a CRT.  But suppose this argument were true (and it is not clear that it is). To restate, assume that there are many functions out there that are uncomputable which map to noise.  

Now consider that most oservations in the natural world are noise.  Any scientific measurement is usually divided into signal (which is rare and what one is usually after) and noise (which is everything else by definition).  Brown noise (which we can define here as $1/f^n, \; n > 0$) is the most common but others are found as well.  Noise is usually discarded but is ever-present.

Therefore if uncomputable functions map to noise, and if noise is by far the most common observation in the natural world, it is implied that uncomputable functions map to most observations of the natural world.  This does not, of course, mean that nothing can be learned about the world using computable functions.  But it does imply that it is a bad idea to disregard uncomputable functions as unimportant in real life.
















