## The three body problem

### Newtonian mechanics

When I was learning Newtonian mechanics in school, I thought that the subject is an open and shut matter: as the laws of motion have been found, the only thing left to do is to apply them to the heavenly bodies to know where they will go.  I was told that this is the case: indeed, consulting the section on Newtonian mechanics in Feynman's lectures on physics (vol. 1 section 9-9), we find this confident statement:

"Now armed with the tremendous power of Newton's laws, we can not only calculate such simple motions but also, given only a machine to handle the arithmetic, even the tremendously complex motions of the planets, to as high a degree of precision as we wish!"

This statement seems logical at first glance.  The most accurate equations of force, momentum, gravitational acceleration etc. are all fairly simple and for most examples that were taught to me, there were solutions that does not involve time at all.  We can define the differential equations for which there exists a closed (non-infinite, or in other words practical) solution that does not contain any reference to time as 'solved'.  The mechanics I learned were problems that were solveable with some calculus, usually by integrating over time to remove that variable.  

If one peruses the curriculum generally taught to people just learning mechanics, a keen eye might spot something curious: the systems considered in the curriculum are all systems of two objects: a planet and a moon, the sun and earth.  This is the problem Newton inherited from Kepler, and the solution he found is the one we learn about today. But what about 3 bodies, or more?  Newton attempted to find a similar solution to this problem but failed.  This did not deter others, and it seems that some investigators such as Laplace were confident of a solution being just around the corner, even if they could not find one themselves.

### Poincare and sensitivity to initial conditions

After a long succession of fruitless attempts, Bruns and Poincare showed that the three body problem does not contain a solution approachable with the method of integration used by Newton to solve the two body problem. No one could solve the three body problem, that is make it into a self-contained algebraic expression, because it is impossible to do so!  

Is any solution possible?  Sundman found that there is an infinite power series that describes the three body problem, and in that sense there is.  But in the sense of actually predicting an orbit, the power series is no help because it cannot directly infer the position of any of the three bodies owing to round-off error propegation from extremely slow convergence [ref](https://arxiv.org/pdf/1508.02312.pdf).  From the point of a solution being one in which time is removed from the equation, the power series is more of a problem reformulation than a solution: the time variable is the power series base.  Rather than numerically integrate the differential equations of motion, one can instead add up the power series terms, but the latter option takes an extremely long time to do.  To give an appreciation for exactly how long, it has been estimated that more than $10^{8000000}$ terms are required for calculating the series for one short time step [ref](http://articles.adsabs.harvard.edu/pdf/1930BuAst...6..417B). 

For more information, see Wolfram's notes [here](https://www.wolframscience.com/reference/notes/972d). 

Unfortunately for us, there is no general solution to the three body problem: we cannot actually tell where three bodies will be at an arbitrary time point in the future, let alone four or five bodies.  This is an inversion with respect to what is stated in the quotation above: armed with the power of Newton's laws, we cannot calculate, with arbitrary precision in finite time, the paths of any system of more than two objects.  

### Trajectories of 3 objects are chaotic

Why is there a solution to the two body problem but not three body problem?  One can imagine that a problem with thousands of objects would be much harder to deal with than two objects, but why does adding only one more object create such a difficult problem?

One way to gain an appreciation for why is to simply plot some trajectories.  

In 1914, Poincare observed that "small differences in initial conditions produce very great ones in the final phenomena" (as quoted [here](https://books.google.com/books?id=vGuYDwAAQBAJ&pg=PA271&lpg=PA271&dq=Poincare+3+body+problem+impossibility+1880&source=bl&ots=yteTecRsK8&sig=ACfU3U2ngm5xUXygi-JdLzpU0bwORuOq7Q&hl=en&sa=X&ved=2ahUKEwiO4JT_86zqAhUlZjUKHYn5Dk8Q6AEwDHoECAwQAQ#v=onepage&q=Poincare%203%20body%20problem%20impossibility%201880&f=false). 


### The three body problem is general 

One can hope that the three (or more) body problem is restricted to celestial mechanics, and that it does not find its way into other fields of study.  Great effort has been expended to learn about the orbitals an electron will make around the nucleus of a proton, so hopefully this knowledge is transferrable to an atom with more than one proton. This hope is in vain: any three-dimensional system with three or more objects that operates according to nonlinear equations reaches the same difficulties outlined above for planets. 


### Does it matter that we cannot solve the three body problem, given that we can just 'solve' the problem on a computer?

When one hears about solutions to the three body problem, they are either restricted to a (miniscule) subset of initial conditions or else are references to the process of numerical integration by a computer.  The latter idea gives rise to the sometimes-held opinion that the three body problem is in fact solveable now that high speed computers are present, because one can simply use extremely precise numeric methods to provide a solution.  

To gain an appreciation for why computers cannot solve our problem, let's first pretend that perfect observations were able to be made.  Would we then be able to use a program to calculate the future trajectory of a planetary system exactly?  We have seen that we cannot when small imperfections exist in observation, but what about if these imperfections do not exist?  Even then we cannot, because it appears that Newton's gravitational constant, like practically all other constants, is an irrational number.  This means that even a perfect measurement of G would not help because it would take infinite time to enter into a computer exactly.



