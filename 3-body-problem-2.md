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

```c++
int main(void){
  ...
  double *p1_x, *p1_y, *p1_z;
  ...
  double *p1_prime_x, *p1_prime_y, *p1_prime_z;
  ...
  double *dv_1_x, *dv_1_y, *dv_1_z;
  ...
```

```c++
int main(void){
  ...
  double *d_p1_x, *d_p1_y, *d_p1_z;
```

```c++

  p1_x = (double*)malloc(N*sizeof(double));
  ...
  still_together = (bool*)malloc(N*sizeof(bool));
  times = (int*)malloc(N*sizeof(int));
  not_diverged = (bool*)malloc(N*sizeof(bool));  
```

```c++
  for (int i = 0; i < N; i++) {
    int remainder = i % resolution;
    int step = i / resolution;
    p1_x[i] = -20. + 40*(double(remainder)/double(resolution));
    p1_y[i] = -20. + 40*(double(step)/double(resolution));
    ...
```

```c++
  cudaMemcpy(d_p1_x, p1_x, N*sizeof(double), cudaMemcpyHostToDevice);
```

```c++
std::chrono::time_point<std::chrono::system_clock> start, end;
start = std::chrono::system_clock::now();
divergence<<<(N+255)/256, 256>>>(
      N, 
      steps, 
      delta_t,
      d_still_together,
      d_not_diverged,
      d_times,
      m1, m2, m3,
      critical_distance,
      d_p1_x,
      ...
      );
  // don't proceed until kernal run is complete
  cudaDeviceSynchronize();
  // measure elapsed kernal runtime
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "Elapsed Time: " << elapsed_seconds.count() << "s\n";
```

```c++
/ kernal declaration
__global__
void divergence(int n, 
              int steps,
              double delta_t,
              bool *still_together,
              bool *not_diverged,
              int *times,
              double m_1, double m_2, double m_3,
              double critical_distance,
              double *p1_x, double *p1_y, double *p1_z, 
              double *p1_prime_x, double *p1_prime_y, double *p1_prime_z, 
              double *dv_1_x, double *dv_1_y, double *dv_1_z,
              double *dv_1pr_x, double *dv_1pr_y, double *dv_1pr_z,
              double *v1_x, double *v1_y, double *v1_z,
              double *v1_prime_x, double *v1_prime_y, double *v1_prime_z,
              double *nv1_x, double *nv1_y, double *nv1_z,
              double *nv1_prime_x, double *nv1_prime_y, double *nv1_prime_z,
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  for (int j=0; j < steps; j++) {
    if (i < n){
      // compute accelerations
      dv_1_x[i] = -9.8 * m_2 * (p1_x[i] - p2_x[i]) / pow(sqrt(pow(p1_x[i] - p2_x[i], 2) + pow(p1_y[i] - p2_y[i], 2) + pow(p1_z[i] - p2_z[i], 2)), 3) \
                  -9.8 * m_3 * (p1_x[i] - p3_x[i]) / pow(sqrt(pow(p1_x[i] - p3_x[i], 2) + pow(p1_y[i] - p3_y[i], 2) + pow(p1_z[i] - p3_z[i], 2)), 3);
      dv_1pr_x[i] = -9.8 * m_2 * (p1_prime_x[i] - p2_prime_x[i]) / pow(sqrt(pow(p1_prime_x[i] - p2_prime_x[i], 2) + pow(p1_prime_y[i] - p2_prime_y[i], 2) + pow(p1_prime_z[i] - p2_prime_z[i], 2)), 3) \
                    -9.8 * m_3 * (p1_prime_x[i] - p3_prime_x[i]) / pow(sqrt(pow(p1_prime_x[i] - p3_prime_x[i], 2) + pow(p1_prime_y[i] - p3_prime_y[i], 2) + pow(p1_prime_z[i] - p3_prime_z[i], 2)), 3);

      // find which trajectories have diverged and increment *times
      not_diverged[i] = sqrt(pow(p1_x[i] - p1_prime_x[i], 2) + pow(p1_y[i] - p1_prime_y[i], 2) + pow(p1_z[i] - p1_prime_z[i], 2)) <= critical_distance;
      still_together[i] &= not_diverged[i];
      if (still_together[i] == true){
        times[i]++;
      };

      // compute new velocities
      nv1_x[i] = v1_x[i] + delta_t * dv_1_x[i];
      nv1_prime_x[i] = v1_prime_x[i] + delta_t * dv_1pr_x[i];

      // compute positions with current velocities
      p1_x[i] = p1_x[i] + delta_t * v1_x[i];
      p1_prime_x[i] = p1_prime_x[i] + delta_t * v1_prime_x[i];

      // assign new velocities to current velocities
      v1_x[i] = nv1_x[i];

      v1_prime_x[i] = nv1_prime_x[i];
      }
    }
  }
```

The same needs to be done for all `x, y, z` vectors of `p1, p2, p3` in order to track all the necessary trajectories.  In total we have 63 vectors to keep track of, which makes the cuda code somewhat unpleasant to write even with the help of developer tools.

The cuda kernal with driver c++ code can be compiled via `nvcc`, which is available through the nvidia cuda toolkit.  Linux users be warned that the drivers necessary for full nVidia toolkit use with an Ampere architecture GPU (such as the author's rtx3060) may not be compatible with the latest kernal version, so downgrading to an older kernal version may be necessary.

Here we compile with the flag `-o` tfollowed by the desired file name where the compiled binary program will be stored.

```bash
(base) bbadger@pupu:~/Desktop/threebody$ nvcc -o divergence divergence.cu
```

For a 300x300 x,y resolution after the full 50,000 timesteps, we have a somewhat disappointing runtime of 

```bash
(base) bbadger@pupu:~/Desktop/threebody$ ./divergence
Elapsed Time: 144.3s
```

Compare this to the `torch` library version of the same problem, which

```python
[Finished in 107.1s]
```

Of course, `torch` itself optimizes the cuda code to some extent so this difference is not particularly surprising and only indicates that the movement of the loop into the cuda kernal does not offer the same performance benefit as other optimizations one can perform.  In the next section we explore methods to optimize the cuda kernal further to achieve faster runtimes for the three body problem than are available using pytorch.

### Optimizing the Three Body Trajectory Computations

Contrary to what one might expect, many parallelized programs applied to GPUs spend more clock cycles (and therefore total time) on memory management than actual computation.  This is nearly always true for deep learning and also holds for many more traditional graphics applications as well.  Memory management occurs both within the GPU and in transfers of data to and from the CPU.  For the three body simulation, a quick look at the code suggests that this program should spend very little time sending data to and from the GPU: we allocated memory for each array, initialized each one before sending to the GPU once, and then copied each array back to the CPU once the loop completes.  This can be confirmed by using a memory profiler such as Nsight-systems, which tells us that the memory copy from GPU (device) to CPU (host) for the 300x300 example requires only ~20ms.  From the screenshot below, it is clear that nearly all the GPU time is spent simply performing the necessary computations (blue boxes on top row).

![profile]({{https://blbadger.github.io}}/3-body-problem/nvidia-nsight.png)

Ignoring GPU-internal memory optimization for the moment, some experimentation can convince us that by far the most effective single change is to forego use of the `pow()` cuda kernal operator for simply multiplying together the necessary operands.  The reason for this is that the cuda `pow(base, exponent)` is designed to handle non-integer `exponent` values which make the evaluatation a [transcendental function](https://forums.developer.nvidia.com/t/register-usage-of-pow/23104), which on the hardware level naturally requires many more registers than one or two multiplication operations.

Thus we can transform the computation of the acceleration into

```c++
  dv_1_x[i] = -9.8 * m_2 * (p1_x[i] - p2_x[i]) / (sqrt((p1_x[i] - p2_x[i])*(p1_x[i] - p2_x[i]) + (p1_y[i] - p2_y[i])*(p1_y[i] - p2_y[i]) + (p1_z[i] - p2_z[i])*(p1_z[i] - p2_z[i]))*sqrt((p1_x[i] - p2_x[i])*(p1_x[i] - p2_x[i]) + (p1_y[i] - p2_y[i])*(p1_y[i] - p2_y[i]) + (p1_z[i] - p2_z[i])*(p1_z[i] - p2_z[i]))*sqrt((p1_x[i] - p2_x[i])*(p1_x[i] - p2_x[i]) + (p1_y[i] - p2_y[i])*(p1_y[i] - p2_y[i]) + (p1_z[i] - p2_z[i])*(p1_z[i] - p2_z[i]))) -9.8 * m_3 * (p1_x[i] - p3_x[i]) / (sqrt((p1_x[i] - p3_x[i])*(p1_x[i] - p3_x[i]) + (p1_y[i] - p3_y[i])*(p1_y[i] - p3_y[i]) + (p1_z[i] - p3_z[i])*(p1_z[i] - p3_z[i]))*sqrt((p1_x[i] - p3_x[i])*(p1_x[i] - p3_x[i]) + (p1_y[i] - p3_y[i])*(p1_y[i] - p3_y[i]) + (p1_z[i] - p3_z[i])*(p1_z[i] - p3_z[i]))*sqrt((p1_x[i] - p3_x[i])*(p1_x[i] - p3_x[i]) + (p1_y[i] - p3_y[i])*(p1_y[i] - p3_y[i]) + (p1_z[i] - p3_z[i])*(p1_z[i] - p3_z[i])));
```

likewise, we can remove the `pow()` operator from our divergence check by squaring both sides of the $L^2$ norm equation

$$
N = \sqrt{x^2_1 + x^2_2 + ... + x^2_n} \\
N^2 = {x^2_1 + x^2_2 + ... + x^2_n} 
$$

which is implemented as

```c++
not_diverged[i] = (p1_x[i]-p1_prime_x[i])*(p1_x[i]-p1_prime_x[i]) + (p1_y[i]-p1_prime_y[i])*(p1_y[i]-p1_prime_y[i]) + (p1_z[i]-p1_prime_z[i])*(p1_z[i]-p1_prime_z[i]) <= critical_distance*critical_distance;
```

other small optimizations we can perform are to change the evaluation of `still_together[i]` to a binary bit check

```c++
if (still_together[i] == 1){
        times[i]++;
      };
```

and the like. With these optimizations in place, we have for the 300x300 example

```bash
(base) bbadger@pupu:~/Desktop/threebody$ ./divergence
Elapsed Time: 44.9377s
```

which is a ~2.4x speedup compared to the `torch` code.  

### Data type optimization

The present optimizations revolve around the 













