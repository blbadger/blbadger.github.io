## Quantum mechanics 

Delayed choice, nonlocality, and more observations from small objects are considered here.  This commentary should be considered as provisional, to be taken with a grain of salt.

### Wave behavior as nonlinearity

Additivity is necessary for linearity.  If bb-gun is fired through two slits onto a wall, the distribution of probabilities that particular areas on the wall that are hit by bbs that pass through the first slit $P_1$ and the second slit $P_2$ is 

$$
P_{12} = P_1 + P_2
$$

or in other words, the slits are additive.  But now consider waves passing through these same slits. In this case, 

$$
P_{12} \neq P_1 + P_2
$$

meaning that adding a second slit to the first is not additive.  More precisely, the amplitude of a wave traveling through the first slit, denoted as the complex number $h_1$, is the real part of $h_1e^{i \omega t}$.  The intensity distribution along the wall with only slit 1 open is $I_1 = \vert I_1 \vert^2$ and only slit 2 open is $I_2 = \vert h_2 \vert^2$, but the intensity distribution with both slits open is

$$
I_{12} = \vert h_1 + h_2 \vert^2 \neq  \vert h_1 \vert^2 + \vert h_2 \vert^2 \\
I_{12} = I_1 + I_2 + 2\sqrt{I_1I_2}cos \delta
$$

A fundamental feature of quantum mechanics is that when path is known, linear particle behavior is observed and $P_{12} = P_1 + P_2$ but when path is unknown, nonlinear wave behavior is observed and $P_{12} = P_1 + P_2 + 2\sqrt{P_1P_2}cos \delta$.  

Quantum mechanics from first principles extends no further: there is no reason why observation of path leads to particle-like behavior and a lack of path observation leads to wave-like behavior. If it is accepted that small particles are by nature probablistic, this is as far as one can go.  But the similarities of quantum behavior to aperiodic dynamical systems abound (inherently unpredictable future from a finitely precise past, necessity for nonlinear transformations etc.) leading to the possibility that one can better explain first principles of quantum mechanics with the principles of nonlinear dynamics.

### Nonlocality implies a nonlinear space: the observer effect

'Spooky action at a distance', or what is now called quantum entanglement, is the phenomenon why which two particles interact such that in a future time at great distance, what happens to one affects what happens to the other (specifically with regards to quantum spin).  There is no intrinsic reasoning capable of explaining such behavior using vector (linear) space, but if one considers a nonlinear space then the result is implied.  In a space defined by a nonlinear function, a change to one area in the space changes other areas because nonlinear transformations are not additive.  Manipulating a particle (or simply observing it, as by the observer effect these are identical) equivalent to changing a parameter in a nonlinear transformation, which changes all areas in the resulting space simultaneously.

### Renormalization and scale

As noted by Feynmann, quantum particle paths are typically non-differentiable.  Equivalently, quantum paths exist in a fractal dimension greater than 1.  This means that the length of the particle path is infinite, because it increases (apparently indefinitely) as the observation resolution increases.  

This is not the only place where infinite quantities intrude in the study of small objects: the process of calculating field strength for quantum particles, for example, is rife with infinite quantities.  To deal with both issues, the theory of quantum renormalization was introduced to remove such infinities, allowing for calculation to proceed safely.

A necessity for renormalization as well as non-differentiable particle paths imply and are implied by scale invariance.  Scale invariance in turn is typical of nonlinear dynamical systems, and thus it is not a stretch of the imagination to propose that the mechanics of small objects are most accurately defined in terms of nonlinear dynamics. As the Schrodinger equation is a linear partial differential equation, the current formulation is either incomplete or, as is more likely, a linear approximation.

Why would field strength need to be renormalized in the first place?  Consider that in many of the pages on this site, maps between dimensions require explosion to infinity (for example, notice how any plane-filling curve must become infinitely long to cover any finite area).  Now fields are three dimensional, whereas if one takes a particle at face value then it is a one dimensional object.  Generation of the field requires passage between dimensions, specifically from a 0-dimensional point to an 3-dimensional (or 4-dimensional if relativity is taken into account) volume.  Only nonlinear transformations are capable of mapping a 0-dimensional point to a 3-dimensional volume, and 

### Implications 

The above considerations have a number of consequences.  Firstly, they suggest that many questions of particle path and field strength are inherently undecidable for the following reason: aperiodic nonlinear dynamical systems require an infinite number of logical operations to predict an arbitrarily small change.  Such systems are inherently uncomputable and therefore to some extent so are the questions of path and field.  

Some of this has long been recognized (although for different reasons than stated here), and similar concerns prompted the famous comment from Feynman, stated as follows: "Why should it take an infinite amount of logic to figure out what one stinky little bit of space-time is going to do?".  But it should be recognized that the implications continue past this issue.  The presence of an apparent continuum of scale in the physical world (which prompted the above comment) means that physical theories attempt not to predict what will happen to a starting condition, but what approximately will happen to an approximate start.  

But if physical theories at all scales behave according to nonlinear transformations that are aperiodic, then arbitrarily small changes to starting values lead to unpredictable predictions.  What this means is that the accuracy for predicting such attributes as bond angle from orbitals and spin changing etc. rely on a nonlinear system to be periodic. 

To see just how unlikely it is that a nonlinear physical system is periodic for more than two objects, consider the classical [three-body problem](https://blbadger.github.io/3-body-problem.html) for celestial mechanics.  Nearly all possible orientations of three bodies in space yield unpredictable outputs (if bounded) from nonlinear gravitational partial differential equations.  The inability for current quantum mechanical systems to predict events with more than three objects is implied by nonlinearity.  

Furthermore, the similarity between nondifferentiable motion in quantum particles and that observed for Brownian motion (see [here](https://blbadger.github.io/additivity-order.html) for more on Brownian motion) naively suggests a similar physical cause.  This implies that just as the particles undergoing Brownian motion are far larger than the water molecules causing this motion, there may well be particles much smaller than currently observed that buffet about the quantum bodies that are observed.  In this sense, there may be no quantum scale at all, merely a relatively quantum scale for our observations because size continues to shrink indefinitely.

If this seems fantastic, consider photons from a point source traveling through a vacuum to land on a photographic plate.  The resolution limit of a blue photon is on the order of 200nm, meaning that the photons land in a blob of about this diameter called an airy disk.  Why do the photons not simply make a point on the plate?  The current account for such behavior is the wave nature of light, and of all small particles.  But observe that the intensity of the central region is a normal distribution, precisely the same as observed for a collection of particles in a fluid undergoing [Brownian motion](https://en.wikipedia.org/wiki/Brownian_motion).  The longer the wavelength of the light, the larger the blob and equivalently the longer the particles diffuse in fluid.  Flight length of a photon has no effect on distribution width because photons experience infinite time dilation.

### Example: electron orbitals and period doubling

A concrete example for how nonlinearity could be important for quantum mechanical systems is in atomic orbitals.  Consider that larger energies are required to attain each orbital in succession s, p, d, f and that the maximum number of locations (physically speaking, the distinct clouds) an electron can exist in for a given m value is, respectively, 1, 2, 4, 8.  Whether or not the period doubling here follows Feigenbaum's $\delta$ is unclear.

Now note that electron orbitals may be considered as the periodicity of electrons: one cloud corresponds to period one etc. Period doubling is typical of nonlinear systems upon an increase in energy, and therefore observed orbital geometry is consistent with period doubling.  If an electron absorbs enough energy, it is discharged from an orbital and travels freely, and most accurately in a non-differentiable path.  This is an aperiodic path with a fractal dimension greater than one, meaning that the process of energy addition to a bound electron (eventually freeing the particle) is consistent with that for a nonlinear attractor.  

### The uncertainty principle

The uncertainty principle states that it is impossible to know a particle's exact momentum as well as position, as the standard deviation in position $\sigma_x$ and standard deviation in momentum $\sigma_p$ are related to Planck's constant $h$
by 

$$
\sigma_x \sigma_p \geq \frac{h}{4 \pi}
$$

which follows from the wave-like nature of small objects, and also applies to macroscopic waves.  Equivalently, one cannot define a unique wavefunction for a very short waveform.  As precision in location increases as the waveform length decreases, necessarily there is a limit to the precision of the accuracy of location and momentum (which is defined by the wavefunction).

But note that this only applies to predictions of the future, not observations in the past.  A photon passing through a double slit, for example, can be determined precisely in both location and momentum after the fact, that is, after it is no longer a wave.  

The result is that small objects (which have little mass) resemble 3-dimensional probability clouds rather than one-dimensional points.  But this is only true for future predictions rather than events that happened in the past, which implies a transformation from a 0-dimensional

### Aside: Delayed choice

The celebrated models of this study such as Schrodinger's equation imply a particle-wave duality of such objects.  Such a duality was observed upon experimentation with light, which in some experimental apparatus behave as though they were particles and in other experiments behave as though they were waves.  An example of the former case is the excitation of electrons to discrete energy levels, and an example of the latter is the dual slit experiment in which a photon interferes with itself, as a macroscopic wave does, when travelling through two closely spaced slits.  Wave-like or particle-like behavior may be observed from objects such as photons or electrons (or even much larger things) but never both at the same time: this is the complementarity principle.

Wheeler devised a thought experiment, now known as the delayed choice, to test complementarity.  It proceeds as follows: suppose gravitational lensing were used to make a cosmic inferometer, which is a method for telling how far away something is.  With that information, it is possible to know the path a photon took as it travelled between galaxies.  Wheeler's hypothesis was that as a photon is neither a particle nor a wave until observed, but instead something indeterminate, then the act of observing the path choice would preclude a wave-like behavior manifest by

This experiment has been carried out, though not as Wheeler initially proposed.  In various experiments, it has been shown that the method used to observe a photon at the end of its path through a double slit with inferometer determines its wave or particle behavior in the entire apparatus, even through the double slit itself.  

One way to think about this is to try to understand when the photon 'learns' the experimental apparatus.  Now note that the Lorenz time dilation for a photon is infinite because it travels at the speed of light:

$$
\Delta t' = \frac{\Delta t}{1-\frac{v^2}{c^2}} \implies \\
\Delta t' \to \infty \; as \; v \to c 
$$

which means that from the photon's perspective, all events are simultaneous.  

If this is accepted, the question of when the photon learns the apparatus is moot because the ending time is equal to the start time.  A change in the method of recording the photon at the end of a photon flight can change its behavior through a double slit 20ns beforehand (in our reference).  Delayed choice and quantum erasure would be more accurately thought of as a necessary result of simultaneity from a photon's perspective.  This idea also accounts for the finding that a photon can be influenced after detection, ie after it is destroyed.

### Principle of least time as implied by delayed choice

Light travels at different speeds throught different materials.  For example, light travels through water (2.25E8 m/s)  slower than it does in vacuum (3E8 m/s). Fermat noted that light travels along paths in different media of refraction as if it chose the quickest way to get from point A to point B.  Suppose that a new transparent material were invented and light was passed through it. When does light 'learn' what angle it should deflect to in order to minimize the time passing through this object? 










