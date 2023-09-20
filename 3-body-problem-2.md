## Three body problem II: Simulating Divergence with Parallelized Computation

This page is a continuation from [Part 1](https://blbadger.github.io/3-body-problem.html), where simulations of the three body problem are explored.  Here we explore the computation of simulated trajectories using a parallelized computer achitecture.

### Introduction

Most nonlinear dynamical systems are fundamentally irreducable: one cannot come up with a computational procedure to determine the parameters of some object at any given time in the future using a fixed amount of computation.  This means that these systems are inherently sequential to some extent. This being the case, there are still many problems that benefit from computations that do not have to proceed in sequence.  One particular example is the problem of finding which positions in a given plane are stable for the trajectory of three bodies in space.  This problem can be approached by determining the stability of various starting locations in sequence, but it is much faster to accomplish this goal by determining the stabilities at various starting locations in parallel.  In [Part 1](https://blbadger.github.io/3-body-problem.html) this parallel computation was performed behind the scenes using the Python `torch` library, which abstracts away the direct computation of tensors on parallelized computational devices like graphics processing units (GPUs) or tensor processing units.

Even with the use of the `torch` library, however, computation of stable and unstable locations takes a substantial amount of time.

```c++
#include <stdio.h>
#include <iostream>
#include <chrono>

int main(void)
{
  int N = 10000;
  int steps = 50000;
  double delta_t = 0.001;
  double critical_distance = 0.5;
  double m1 = 10;
  double m2 = 20;
  double m3 = 30;
```




