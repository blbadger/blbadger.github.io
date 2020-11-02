## Entropy 

### Introduction: the second law of thermodynamics

The second law of thermodynamics may be stated in may equivalent ways, and one is as follows: spontaneous (and irreversible) processes increase the entropy of the universe.  Another way of saying this is that for irreversible reactions, $S_f > S_i$ or the final entropy is greater than the initial entropy.

Here entropy may be thought of as the amount of disorder.  'Disorder' is a somewhat vague term, and so in statistical mechanics entropy is defined as the number of states a system can exist in, with more possible states corresponding to higher entropy.  Spontaneous processes are those that occur during the passage of time without any external input.  In Gibb's celebrated equation that relates free energy $\Delta G$ (the energy available to do work) to change in enthalpy $\Delta H$ (heat) and entropy $\Delta S$, 

$$
\Delta G = \Delta H - T \Delta S
$$

spontaneous processes are those for which $\Delta G < 0$ for the system.  Therefore, either heat escapes into the surroundings (signified by a negative $\Delta H$) which increases the entropy of the surroundings ($\Delta S = \Delta q / T$), or else entropy increases in the system itself.

The second law may also be stated as follows: the universe becomes more disordered over time. This sort of definitions abounds when entropy is discussed, but before proceeding to examine how accurate they are it is helpful to understand what thermodynamics is, and where entropy came from.  Thermodynamics is the study of heat change over time, just like the name implies.  This study got underway during the 19th century when engineers attempted to make steam engines more efficient, and so the theory of thermodynamics predates atomic theory by many decades.  

The source of the second law is important because it was derived using assumptions which, while useful, are not accurate.  Specifically, the ideal gas law ($PV = nRT$) was used to calculate isothermal gas expansion that was then used to calculate the change in entropy over time ([see here](https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Supplemental_Modules_(Physical_and_Theoretical_Chemistry)/Thermodynamics/The_Four_Laws_of_Thermodynamics/Second_Law_of_Thermodynamics)).  This is important because no gas is actually ideal: real gases all exhibit attractive forces that do not exist in the ideal gas equations.  

Consider the following thought experiment: an exothermic, negatively entropic reaction floating in the cold vacuum of space, billions of light years from any other matter.  Now the standard interpretation is that this reaction produces heat which spreads to the outside universe, thereby increasing total entropy.  So our floating reaction radiates heat into space but does this increase the universal entropy, in the statistical mechanical sense? Not unless the radiation encounters matter, which may never occur. Will the reaction still proceed even though there actually a decrease in universal entropy?   

### A first consideration: fundamental forces and order

Consider the statement "the universe becomes more disordered over time".  Is this true of the cosmos over the timescales we can observe, nearly 14 billion years?  As far as has been observed, it is not: galactic structures (the ones we can see at least) have become less, rather than more, chaotic over time.  ~13.7 billion years is a long time, and it seems strange for the observable universe to appear more ordered over that time scale if indeed every sppontanous process caused the universe to become more disordered.  What stops this from happening?

The spontaneous expansion of gas (that can be used to do work) may be explained using entropy as follows: by increasing the volume each gas molecule can exist in, the number of available states to that molecule increases and so entropy increases. At our scale and temperatures, gas indeed expands.  But not consider our atmosphere: it does not expand, even though it is entropically favorable to do so. The force acting against expansion is gravity, and the cosmos is full of examples of this force acting to prevent and reverse the expansion of gas. Contraction of gas may but usually does not occur isothermally, meaning that gravity acts to restrict molecular motion but usually at the expense of energy dispersion.

At very small scale, the force of electromagnetism confines electrons to exist near nuclei in distinct cloud-like orbitals.  A freely moving electron and proton are confined to become a hydrogen atom by the electromagnetic force if they interact. This interaction will release a photon, which may or may not interact with matter.  The same is true of the strong nuclear force: quarks that potentially move freely are constrained by the strong force.  Once again, these processes act to restrict motion and thereby reduce disorder, but also usually disperse energy in the form of photons (that may or may not in turn come into contact with other matter).

### Second consideration: infinite states 

Work can be defined as directed movement: when one does work to move a log, all molecules in the log move in approximately concerted motion.  In contrast, thermal motion is not directed: heat causes molecules to move faster but not in concerted motion.  In statistical mechanics, the number of internal states (formally, the microcanonical ensemble) that a molecule can assume is used to calculate its entropy.  

$$
S = k \; ln W
$$

where $W$ signifies the number of microstates available in a collection of particles and $k$ is Boltzmann's constant.  

If we restrict ourselves to energy levels alone, it is often thought that quantum mechanics allows us to obtain finite numbers of microstates $W$ and therefore the notion of entropy as being the number of available states is useful (ie a finite quantity).  But it is no longer useful when one considers motion through space: there are an uncountably infinite number of directions in which a gas molecule may move.  And on closer examination of even a simple atom, there seems to be an uncountably infinite number of microstates even in the framework of quantum mechanics: although discrete orbitals and bond angles do exist, real bond angles vary more or less continuously and electron orbitals are only probable locations of electrons, not discrete limits.  

Suppose one accepts that energy levels are purely discrete.  As molecular orientation is important (and the defining characteristic between energy able to do work and heat), molecule that obeys quantum mechanics still has an uncountably infinite number of states at any time.  Defining entropy by the number of microstates is unquestionably useful, but does not appear to be an entirely accurate description of nature.

### Third consideration: combinatorial microstates

Consider the idea that entropy is determined by the number of microstates avaliable to a collection of molecules, with more microstates possible signifying a higher entropy.  The current understanding that the number of microstates is maximized upon a dissipation of head presents a problem: what about the relation of molecules in one ensemble to each other (the combinatorial microstate)?  This is of more than theoretical concern, as many subatomic particles (electrons etc.) are practically indistinguishable if they exist at a similar energy.

To see the problem, say we have 3 particles $x, y, z$ that may each exist in one of 3 possible locations $A, B, C$.  If the particles are identical and exhibit identical thermodynamic properties, we can regard these as identical and name them $x, x, x$: these may be the results of a reaction proceeding to thermodynamic equilibrium.  How many possible combinatorial microstates are there if the particles are identical?  If they can exist in any of $A, B, C$ then there are 3 possible locations each containing 1, 2, or 3 particles: $3 * 3 = 9$ possible microstates:

| A | B | C |
|---|---|---|
| x | x | x |  
|   |  xx | x  |  
|   |   |  xxx |   
$$...$$


But now imagine that the particles are distinguishable: perhaps they are thermodynamically distinct.  Now, the number of possibilities for particles $x, y, z$ is
3 places, three particles, where order matters (meaning that xy at location A is distinct from xz at location A): $3! * 3 = 18$ possible microstates:


| A | B | C |
|---|---|---|
| x | y | z |  
|   |  xy | z  | 
|    |  xz | y|
|   |   |  xyz |  
$$
...
$$

This means that there are more possible microstates available to a particle ensemble where the elements are thermodynamically distinct: in other words, entropy is maximized for ensembles that contain thermodynamically distinct elements. 

For more problems resulting from indistinguishability of small particles, see the Gibbs Paradox (https://en.wikipedia.org/wiki/Gibbs_paradox). 

### Fourth consideration: exceptions to the rule

The first law of thermodynamics is that energy cannot be created nor destroyed, and there are no phenomena that I am aware of that flout this rule.  This is not the same for the second law, and two particular examples of importance will be discussed here.  It should be made clear that nearly every other relevant physical observation supports rather than fails to support the second law, but the existence of counterexamples does suggest that our understanding is presently incomplete. 

A first exception is Brownian motion, the movement of small (<0.1 mm) particles in fluids such as water.  Originally thought to be a biological phenomena (observed with pollen grains), Brown provided evidence that nonliviing matter undergoes Brownian motion too (hence the name).  Einstein used Brownian motion as evidence for the theory of the atom, and it is now understood that this motion is the result of a huge number of molecules undergoing thermal motion that collide with a larger particle, and that these collisions add and subtract and tend to push the particle along an everywhere-nondifferentiable path.  

Why does Brownian motion flout the second law?  Intuitively, because it converts heat to directed motion.  Heat causes random motion in molecules of the surrounding liquid, but the summation of the impacts of these molecules with the larger object lead to motion that is somewhat less than random (although very difficult to predict).  More concretely, Brownian motion in three dimensions is an example of an irreversible, spontaneous process that converts heat to work: irreversible because a particle undergoing Brownian motion does not revisit its initial position an arbitrary number of times in the future (ie it travels away from its initial spot over time), spontanous because no free energy is required, and it converts heat to work because work is equal to force times distance, and it takes a (very tiny) amount of force to move a tiny particle from point a to b in a fluid due to friction.  The energy source for this movement is thermal motion of molecules, meaning that heat is converted to work.  

A second exception is the arrangement of electrons (or any charged particle) on the surface of a three dimensional conductor. This is a spontanous, irreversible process that decreases the number of possible locations of the charged particles relative to what is possible if they were to fill the volume of the conductor.  By the definition of entropy as a measure of possible orientations, the process of particles moving from a state of greatest volume to a state of limited (actually zero) volume implies a decrease in entropy over time.  

Let's look more closely at why: the movement of charged particles to the surface of a conductor is spontanous because it occurs with no external work, it is irreversible because to make the particles not collect on the surface of the conductor does require work, and it decreases entropy because there are far fewer possible locations to exist in on the surface of a solid than in the interior.  As an analogy, imagine that gas molecules collected on the inside surface of a gas chamber: in this case, the molecules occupy a negligeable fraction of the volume they could possibly exist in, and thus have undergone isothermal contraction.  As isothermal expansion is entropically favorable, the opposite is the reverse. 

### Entropy and the second law, more precisely: Energy dissipation

With the considerations of attempting to avoid defining entropy in terms of microstates, or as a measure of disorder, we can try to reformulate the second law of thermodynamics to be as accurate as possible.  And here it is: Energy tends to spread out over time.  What does this mean for physical systems? Phase space dissipation is the process by which some area of phase space is reduced to measure 0 over time in a dynamical system.  

In other words, the universe acts as a dissipative dynamical system with regards to energy, at least most of the time.  Now consider this: dissipative but not conservative (conservative meaning that energy does not spread out over time)  dynamical systems may have fractal attractors (for a good review on this topic, see [here](https://www.osapublishing.org/josaa/abstract.cfm?uri=josaa-7-6-1055). 

From this perspective, it is perhaps not difficult to see why there are so many fractal objects in nature: dissipation is necessary for self-similar phase space structures to form.  Phase space is an abstract mathematical system for describing dynamics and is thus not directly related to physical shape, but it is important to note that many dynamical systems that are self-similar over time (in phase space) are also self-similar in structure at one time: turbulence is the classic example of this phenomenon.  

All this leads to an interesting conclusion: the second law of thermodynamics implies that energy usually spreads out which leads to the formation of self-similar fractal behavior over time structures in space.  If we define entropy as dissipation of energy and the formation of fractals as decreasing the possible number of states (as behavior at very different scales is similar in some way or another), then the conclusion is that the second law of thermodynamics implies that energy spreads out, increasing entropy, which decreases the number of possible states and thus decreases disorder.  We have performed an inversion with respect to the effect of the second law on order.







