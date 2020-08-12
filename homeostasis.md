## Homeostasis

### Background

Homeostasis is a fundamental feature of living organisms, and is defined by [Britannica](https://www.britannica.com/science/homeostasis) to be 

"Any self-regulating process by which biological systems tend to maintain stability while adjusting to conditions that are optimal for survival"

although the conditions to which a biological system is subjected to are usually actually sub-optimal.  The next sentence is:

"If homeostasis is successful, life continues; if unsuccessful, disaster or death ensues." 

The idea here is that if a living organism is moved from one environment to another, it must change its physiology such that some internal internal parameters are unchanged, or else the organism dies.  Exactly which parameters are important depends on the organism: warm-blooded endothermic animals cannot tolerate more than a few degrees of core temperature change, whereas ectotherms can tolerate a wide range of temperatures while still maintaining homeostasis and thefore stability (ie little change) in other parameters.  

Dramatic demonstrations of homeostasis have been made to audiences: in one of the most famous, the physician Blagden entered an oven with a dog and a piece of meat, and emerged forty minutes later with the dog and a cooked steak.  Blagden and his dog cooled themselves via evaporation of water, and were able to maintain homeostasis whereas the meat could not, and therefore it was irrevocably changed.

The connection between homeostasis and life is nearly always present: organisms that fail to undergo homeostasis die.  The exceptions are the dormant stages of life that commonly occur upon dessication or freezing or germination.  

Continuing our definition:

"The stability attained is actually a dynamic equilibrium, in which continuous change occurs yet relatively uniform conditions prevail."

Dynamic means changing over time, therefore homeostasis exists in a dynamical system.  The continuous changes that result in relatively uniform internal conditions can be thought of as the path, or trajectory, of the dynamical system.  Here the change in parameters of temperature, oxygen consumption etc. may be plotted over time in the same way that a phase space may be used to observe the changes in kinetic energy as position changes for a [pendulum](/pendulum-map.md).

### Homeostasis as an attractor in a nonlinear dynamical system

What type of dynamical system would lead to continuous changes that preserve similar conditions over time, even when outside influences nudge them towards instability?  Any organism that ages does not re-enter precisely the same condition as it was in before, otherwise we could remove time from the system and the organism would last perpetually without some adverse outside influence.  But for nearly all organisms, this is not the case and ageing occurs, which can eventually disrupt homeostasis once and for all. 

This means that homeostasis is not just in any dynamical system: it is an [aperiodic dynamical system](/index.md), or one that does not precisely revisit a previous state as time passes.  As only nonlinear (or piecewise linear, which one can think of as being discontinuously nonlinear) dynamical systems exhibit attractors that are not points (with measure > 0) and are periodic, homeostasis can be viewed as an attractor in a nonlinear dynamical system.

Another line of reasoning reaches the same conclusion.  Negative feedback is feedback that is used to decrease the distance from a desired state.  The classic example here is a home thermostat: if the temperature is too low, a heater is activated until it is within an acceptable margin.  If the temperature is too high, cooling is activated and both heating and cooling are negative feedback. 

Homeostasis can be equated to the combination of negative feedback events.  Feedback can be stable, where Negative feedback can be linear, in which case it yields a stable or unstable outcome regardless of the inputs, or else it can be nonlinear in which case some inputs can yeild stability and some yield instability (see Pierce 1961).   The many instances of homeostatic feedback that takes a small and relatively harmless input and yields an undesirable output (for example take any allergy or the temperature fall during hypothermia, etc.) means that the feedback is not stable regardless of the inputs.  Thus the feedback must be nonlinear, meaning that homeostasis exists as an attractive state in a nonlinear system. 


### Implications for biological research

Why does it matter if homeostasis is a nonlinear dynamical attractor?  It matters because nonlinear systems are not additive, and the scientific method of experimentation via variable isolation and manipulation assumes linearity (specifically additivity).  This means that nonlinear systems are poorly understood by the methods which are normally used for biological research.

To see why experimentation assumes linearity, imagine you want to understand how a mechanical watch works.  You open it up and see a little wheel spinning to and fro, many gears moving at various speeds, and a spring.  What does the spring do?  Remove it and all the gears stop: therefore the spring is necessary to cause the gears (and ultimately the hands) to move.  In mathematical logic terms, 'if $X$ then $Y$' or formally $ X \implies Y $ means the same thing as '$Y$ is necessary for $X$' and so for our watch, if it is moving ($X$) then it must have a spring ($Y$).

In order to learn this, the spring alone must be affected.  If removing the spring also causes all the gears to fall out then the conclusion above is no longer necessarily true.  Changing only the spring is an isolation of variables, and it is necessary for an experiment to yield any information.  But now imagine that the watch is nonlinear: the parts are not additive, meaning that they are not separable.  This means that whenever someone tries to remove the spring, the gears fall out.  Then the conclusion is that the spring is necessary to stop the watch hands from spinning freely could be made.  This is not accurate because it is actually the gears that prevent the hands from spinning freely. By failing to isolate the spring variable from the gear variable, a conclusion based on experimentation is not helpful.

The above example may seem silly at first glance but consider that an analagous process occurs in any homeostatic organism.  Say we want to understand the function of protein X in a cell, so we induce some change to reduce it's number in the cell by a fifth.  Without knowning which protein this is, a molecular biologist could reasonably predict that nothing will happen.  Indeed, the majority of proteins in the cell are non-essential meaning that if all of one of these is removed, homeostasis remains.  Does this mean that none of these proteins do anything in the cell?  No, because the variables are not separable: removing one changes the rest, in this case perhaps causing other proteins to compensate for the loss of one. 

The current reproducibility crisis of scientific fields investigating living organisms (biomedicine, psychology etc.) is likely a consequence of attempts to apply a method that assumes linearity to systems that are not linear.  Nonlinearity means that isolation of variables is not possible, and therefore experimentation leading to meaningful prediction is not possible for an arbitrary nonlinear system.


### Approximations 

Does nonlinearity always matter? In other words, can we ever study a nonlinear system by pretending that it is linear and using the scientific method of variable isolation and experimentation?  Sometimes yes, when any particular influence is large enough.  This is the case for the [three body problem](/three-body-problem.md), a nonlinear system of three astronomical bodies that are subject to the gravitational attraction of each other.  If one of the bodies is much less massive than the others, the three body problem can be approximated by the much simpler two body problem. 

In a similar way, one can draw conclusions about a living system when one or a few factors act to negate homeostasis in some way.  For example, removal of oxygen prevents respiration and kills nearly all animal cells because homeostasis is lost.  We can with reasonable confidence state that oxygen is necessary for respiration because these events cease completely when oxygen is removed.  We can approximate the system as a linear one with respect to oxygen removal because any attempt in a cell to cope with a lack of oxygen (and there are many such attempts) has little effect.

It should be noted that this is only true for large effects that destroy homeostasis in some way, and even then approximations should be treated with care for anything other than extremely short-term prediction. Of course, with homeostasis lost there is likely no long term to think of.




























