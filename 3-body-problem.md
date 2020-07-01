## The three body problem

### Newtonian mechanics

When I was learning Newtonian mechanics in school, I thought that the subject is an open and shut matter: as the laws of motion have been found, the only thing left to do
is to apply them to the heavenly bodies to know where they will go.  I think I thought that was true because I was told that this is the case: indeed, consulting the section
on Newtonian mechanics in Feynman's lectures on physics (vol. 1 section 9-9), we find this confident statement:

"Now armed with the tremendous power of Newton's laws, we can not only calculate such simple motions but also, given only a machine to handle the arithmetic, even
the tremendously complex motions of the planets, to as high a degree of precision as we wish!"

This is the sort of thought that I would have agreed to without question.  The most accurate equations of force, momentum, gravitational acceleration etc. are all fairly simple 
and for most examples that were taught to me, there were solutions that does not involve time at all.  We can define the differential equations for which there exists
a closed (non-infinite, or in other words practical) solution that does not contain any reference to time as 'solved'.  The mechanics I learned were problems that
were solveable with some calculus, usually by integrating over time to remove that variable.  

If one peruses the curriculum generally taught to people just learning mechanics, a keen eye might spot something curious: the systems considered in the curriculum
are all systems of two objects: a planet and a moon, the sun and earth.  



### The three body problem is general 

One can hope that the three (or more) body problem is restricted to celestial mechanics, and that it does not find its way into other fields of study.  Great
effort has been expended to learn about the orbitals an electron will make around the nucleus of a proton, so hopefully this knowledge is transferrable to an 
atom with more than one proton. This hope is in vain: any three-dimensional system with three or more objects that operates according to nonlinear equations
reaches the same difficulties outlined above for planets. 


### Does it matter that we cannot solve the three body problem, given that we can just 'solve' the problem on a computer?

When one hears about solutions to the three body problem, they are either restricted to a (miniscule) subset of initial conditions or else are references to
the process of numerical integration by a computer.  The latter idea gives rise to the sometimes-held opinion that the three body problem is in fact solveable
now that high speed computers are present, because one can simply use extremely precise numeric methods to provide a solution.  

To gain an appreciation for why computers cannot solve our problem, let's first pretend that perfect observations were able to be made.  Would we then be able to
use a program to calculate the future trajectory of a planetary system exactly?  We have seen that we cannot when small imperfections exist in observation, but
what about if these imperfections do not exist?  Even then we cannot, because it appears that Newton's gravitational constant, like practically all other constants,
is an irrational number.  This means that even a perfect measurement of G would not help because it would take infinite time to enter into a computer exactly!



