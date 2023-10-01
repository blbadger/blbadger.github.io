## Three body problem II: Simulating Divergence with Parallelized Computation

This page is a continuation from [Part 1](https://blbadger.github.io/3-body-problem.html), where simulations of the three body problem are explored.  Here we explore the computation of simulated trajectories using a parallelized computer achitecture.

### Introduction

Most nonlinear dynamical systems are fundamentally irreducable: one cannot come up with a computational procedure to determine the parameters of some object at any given time in the future using a fixed amount of computation.  This means that these systems are inherently sequential to some extent. This being the case, there are still many problems that benefit from computations that do not have to proceed in sequence.  One particular example is the problem of finding which positions in a given plane are stable for the trajectory of three bodies in space.  This problem can be approached by determining the stability of various starting locations in sequence, but it is much faster to accomplish this goal by determining the stabilities at various starting locations in parallel.  In [Part 1](https://blbadger.github.io/3-body-problem.html) this parallel computation was performed behind the scenes using the Python `torch` library, which abstracts away the direct computation of tensors on parallelized computational devices like graphics processing units (GPUs) or tensor processing units.

Even with the use of the [optimized](https://pytorch.org/tutorials/advanced/cpp_extension.html) torch library, however, computation of stable and unstable locations takes a substantial amount of time.  Most images displayed in [part 1](https://blbadger.github.io/3-body-problem.html) require around 18 minutes to compute: this is due to the large number of iteration required (50,000 or more), the large size of each array (18 arrays with more than a million components each) and even the data type used (double precision 64-bit floating point).

### A CUDA kernal for divergence

Here we will explore speeding up the three body computation by writing our own GPU code, rather than relying on torch to supply this when given higher-level instructions.  The author has an Nvidia GPU and code on this page will therefore be written in C/C++ CUDA (Compute Unified Device Architecure).  The code contains a standard C++ -style library inclusion and function initialization (C++ execution always begins at `int main()`), all of which is performed on the CPU.  Here we first initialize some constants for the three body similation.

```c++
#include <stdio.h>
#include <iostream>
#include <chrono>

int main(void)
{
  int N = 90000;
  int steps = 50000;
  double delta_t = 0.001;
  double critical_distance = 0.5;
  double m1 = 10;
  double m2 = 20;
  double m3 = 30;
```

And then we continue by assigning pointer variables with the proper data type for each of our planets.  This is the most efficient form of an array in C++, allowing us to allocate memory and initialize each element directly.  

Here `N` is the number of pixels, ie a 300x300 divergence plot contains 90,000 pixels. We will perform all the required computations in 1D arrays for now, such that separate arrays are initialized for each x, y, z component of each attribute of each planet.  We have to initialize position, acceleration (dv), velocity, and a temporary buffer array for new velocities.

```c++
int main(void)
{
  ...
  double *p1_x, *p1_y, *p1_z;
  ...
  double *dv_1_x, *dv_1_y, *dv_1_z;
  ...
  double *nv1_x, *nv1_y, *nv1_z;
  ...
  double *v1_x, *v1_y, *v1_z;
```

We also want to initialize boolean arrays for trajectories that have not diverged yet for any given iteration, a boolean array for trajectories that are not diverging right now, and an array to keep track of the iteration in which divergence occurred.

```c++
bool *still_together,*not_diverged;
int *times
```

For each array, we must allocate the proper amount of memory depending on the type of data stored. We have $N$ total elements, so we need to allocate the size of each element times $N$ for each array.

```c++
  ...
  p1_x = (double*)malloc(N*sizeof(double));
  ...
  still_together = (bool*)malloc(N*sizeof(bool));
  times = (int*)malloc(N*sizeof(int));
  not_diverged = (bool*)malloc(N*sizeof(bool));  
```

Next, space is allocated for each array in the GPU. One way to do this is to declare a new pointer variable corresponding to each array (see below for p1_x) before allocating the proper amount of memory for that variable (which must be referenced as it was defined as a pointer.  Each must be different than the name for each corresponding CPU array, and here `d_` is prefixed to designate that this is a 'device' array.

```c++
double *d_p1_x;
cudaMalloc(&d_p1_x, N*sizeof(double)); 
```

Now we need to initialize each array with our starting condition.  As we are working with 1D arrays rather than the 2D arrays, we need to initialize each array to capture 2D information in a single list.  This is similar to how 2D arrays are represented in memory, and should lead to the fastest computation even for an Nvidia GPU which contains built-in 2D and 3D objects.  

![1d illustration]({{https://blbadger.github.io}}/3_body_problem/1d_cuda.png)

This may be implemented using modulo division in which the x parameter is equivalent to the remainder of the division of the total number of elements by the square root of elements, and the y parameter is equal to the integer (floor) division of the number of elements by the square root of elements. Here we also scale each element by the appropriate constants (here to make a linear interpolation between -20 and 20)

```c++
  for (int i = 0; i < N; i++) {
    int remainder = i % resolution;
    int step = i / resolution;
    p1_x[i] = -20. + 40*(double(remainder)/double(resolution));
    p1_y[i] = -20. + 40*(double(step)/double(resolution));
    ...
```

Now we copy each array from the CPU to the GPU

```c++
  cudaMemcpy(d_p1_x, p1_x, N*sizeof(double), cudaMemcpyHostToDevice);
```

and now we can run the CUDA kernal, keeping track of the time spent.  
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

```

CUDA functions are termed 'kernals', and are called by the kernal name followed by the number of grid blocks and threads per block for execution. We call our CUDA function by `divergence<<<blocks, threads_per_block>>>(args)`. The denominator of the `blocks` must equal the `threads_per_block` for this experiment for reasons detailed below.

We have to synchronize the GPU before measuring the time of completion, as otherwise the code will continue executing in the CPU after the kernal instructions have been sent to the GPU.

```c++
  // don't proceed until kernal run is complete
  cudaDeviceSynchronize();
  // measure elapsed kernal runtime
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "Elapsed Time: " << elapsed_seconds.count() << "s\n";
```

A typical CUDA kernal declaration is `__global__ void funcname(args)` although `__device__` may also be used.  For brevity, only the first planet's arrays are included below but note that the full kernal requires all three planet arrays.

```c++
// kernal declaration
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
)
```
Parallelized computation must now be specified: in the following code, we define index `i` to be a certain block's thread, in one dimension as this is how the arrays were defined as well. Note that as the array is 1D, `blockDim.x` will always evaluate to 1.  The arrangement of blocks and threads in our kernal call is now clearer, as each thread is responsible for one index.

```c++
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
```

Now the trajectory simulation computations are performed. In the spirit of refraining from as much data transfer from the CPU to the GPU and back, we will perform the simulation calculations entirely inside the GPU by moving the trajectory loop to the CUDA kernal.  This is notably different than the pytorch approach, where the loop existed on the CPU side (in python) and the GPU was instructed to perform one array computation at a time. It can be shown that moving the loop to the GPU saves a small amount of time, although less than what the author would expect (typically ~5% of runtime).

For each index `i` corresponding to one CUDA thread, $steps$ iterations of the three body trajectory are performed and at each iteration the $times$ array is incremented if the trajectory of planet one has not diverged from its slightly shifted counter part (planet one prime). 

```c++
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

```

The same needs to be done for all `x, y, z` vectors of `p1, p2, p3` in order to track all the necessary trajectories.  In total we have 63 vectors to keep track of, which makes the cuda code somewhat unpleasant to write even with the help of developer tools.

The cuda kernal with driver c++ code can be compiled via `nvcc`, which is available through the nvidia cuda toolkit.  Linux users be warned that the drivers necessary for full Nvidia toolkit use with an Ampere architecture GPU (such as the author's rtx 3060) may not be compatible with the latest kernal version, so downgrading to an older kernal version may be necessary.

Here we compile with the flag `-o` followed by the desired file name where the compiled binary program will be stored.

```bash
(base) bbadger@pupu:~/Desktop/threebody$ nvcc -o divergence divergence.cu
```

For a 300x300 x,y resolution after the full 50,000 timesteps, we have a somewhat disappointing runtime of 

```bash
(base) bbadger@pupu:~/Desktop/threebody$ ./divergence
Elapsed Time: 144.3s
```

Compare this to the torch library version of the same problem, which

```python
[Finished in 107.1s]
```

Pytorch employs some optimizations in its CUDA code, so this difference is not particularly surprising and only indicates that the movement of the loop into the cuda kernal does not offer the same performance benefit as other optimizations that are possible.  In the next section we explore methods to optimize the cuda kernal further to achieve faster runtimes for the three body problem than are available using torch.

### Optimizing the Three Body Trajectory Computations

Contrary to what one might expect, many parallelized programs applied to GPUs spend more clock cycles (and therefore total time) on memory management than actual computation.  This is nearly always true for deep learning and also holds for many more traditional graphics applications as well.  Memory management occurs both within the GPU and in transfers of data to and from the CPU.  For the three body simulation, a quick look at the code suggests that this program should spend very little time sending data to and from the GPU: we allocated memory for each array, initialized each one before sending to the GPU once, and then copied each array back to the CPU once the loop completes.  This can be confirmed by using a memory profiler such as Nsight-systems, which tells us that the memory copy from GPU (device) to CPU (host) for the 300x300 example requires only ~20ms.  From the screenshot below, it is clear that nearly all the GPU time is spent simply performing the necessary computations (blue boxes on top row).

![profile]({{https://blbadger.github.io}}/3_body_problem/nvidia-nsight.png)

Ignoring GPU-internal memory optimization for the moment, some experimentation can convince us that by far the most effective single change is to forego use of the `pow()` cuda kernal operator for simply multiplying together the necessary operands.  The reason for this is that the cuda `pow(base, exponent)` is designed to handle non-integer `exponent` values which make the evaluatation a [transcendental function](https://forums.developer.nvidia.com/t/register-usage-of-pow/23104), which on the hardware level naturally requires many more registers than one or two multiplication operations.

Thus we can forego the use of the `pow()` operator for direct multiplication in order to optimize the three body trajectory computation.  This change makes a somewhat-tedious CUDA code block become extremely tedious to write, so we can instead have a Python program write out the code for us.  

```python
def generate_string(a: str, b: str, c: str, d: str, t: str, , m: str, prime: bool) -> str:
    if prime:
        e = '_prime'
    else:
        e = ''
    first_denom =  f'sqrt(({a}{e}_x[i] - {b}{e}_x[i])*({a}{e}_x[i] - {b}{e}_x[i]) + ({a}{e}_y[i] - {b}{e}_y[i])*({a}{e}_y[i] - {b}{e}_y[i]) + ({a}{e}_z[i] - {b}{e}_z[i])*({a}{e}_z[i] - {b}{e}_z[i]))' 
    second_denom = f'sqrt(({c}{e}_x[i] - {d}{e}_x[i])*({c}{e}_x[i] - {d}{e}_x[i]) + ({c}{e}_y[i] - {d}{e}_y[i])*({c}{e}_y[i] - {d}{e}_y[i]) + ({c}{e}_z[i] - {d}{e}_z[i])*({c}{e}_z[i] - {d}{e}_z[i]))'
    template = f'''-9.8 * m_{m} * ({a}{e}_{t}[i] - {b}{e}_{t}[i]) / ({first_denom}*{first_denom}*{first_denom}) -9.8 * m_{m} * ({c}{e}_{t}[i] - {d}{e}_{t}[i]) / ({second_denom}*{second_denom}*{second_denom});'''
    return template

a, b = 'p3', 'p1'
c, d = 'p3', 'p2'
m = '3'
t = 'z'
print (generate_string(a, b, c, d, t, prime=True))
```

For the acceleration of planet 1, this gives us

```c++
  dv_1_x[i] = -9.8 * m_2 * (p1_x[i] - p2_x[i]) / (sqrt((p1_x[i] - p2_x[i])*(p1_x[i] - p2_x[i]) + (p1_y[i] - p2_y[i])*(p1_y[i] - p2_y[i]) + (p1_z[i] - p2_z[i])*(p1_z[i] - p2_z[i]))*sqrt((p1_x[i] - p2_x[i])*(p1_x[i] - p2_x[i]) + (p1_y[i] - p2_y[i])*(p1_y[i] - p2_y[i]) + (p1_z[i] - p2_z[i])*(p1_z[i] - p2_z[i]))*sqrt((p1_x[i] - p2_x[i])*(p1_x[i] - p2_x[i]) + (p1_y[i] - p2_y[i])*(p1_y[i] - p2_y[i]) + (p1_z[i] - p2_z[i])*(p1_z[i] - p2_z[i]))) -9.8 * m_3 * (p1_x[i] - p3_x[i]) / (sqrt((p1_x[i] - p3_x[i])*(p1_x[i] - p3_x[i]) + (p1_y[i] - p3_y[i])*(p1_y[i] - p3_y[i]) + (p1_z[i] - p3_z[i])*(p1_z[i] - p3_z[i]))*sqrt((p1_x[i] - p3_x[i])*(p1_x[i] - p3_x[i]) + (p1_y[i] - p3_y[i])*(p1_y[i] - p3_y[i]) + (p1_z[i] - p3_z[i])*(p1_z[i] - p3_z[i]))*sqrt((p1_x[i] - p3_x[i])*(p1_x[i] - p3_x[i]) + (p1_y[i] - p3_y[i])*(p1_y[i] - p3_y[i]) + (p1_z[i] - p3_z[i])*(p1_z[i] - p3_z[i])));
```

Likewise, we can remove the `pow()` operator from our divergence check by squaring both sides of the $L^2$ norm equation

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

and the like. Finally, we can halt the CUDA kernal if a trajectory has diverged. This allows us to prevent the GPU from continuing to compute the trajectories of starting values that
have already diverged, which don't yield any useful information.

```c++
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  for (int j=0; j < steps; j++) {
    if (i < n and still_together[i]){
    ...
```

With these optimizations in place, we have for the 300x300 example

```bash
(base) bbadger@pupu:~/Desktop/threebody$ ./divergence
Elapsed Time: 44.9377s
```

which is a ~2.4x speedup compared to the `torch` code, a substantial improvement.  These optimizations become more effective as the number of iterations increases (and thus the area of the input that has already diverged increases): for example, for $i=90,000$ iterations we have a runtime of 771s for the optimized CUDA kernal but 1951s for the `torch` version (a 2.53x speedup) and for $i=150,000$ we have 1095s for our CUDA kernal but 3257s for the torch version.  As the CUDA kernal is executed block-wise such that the computation only halts if all $i$ indicies for that block evaluate to `false`, decreasing the block size (and concomitantly the number of threads per block) in the kernal execution configuration can lead to modest speedups beyond what is reported here.

In the case of block and thread size of 1, the following depicts the difference between our early stopping CUDA code and the torch-based method employed in [part 1](https://blbadger.github.io/3-body-problem.html).  

![early stopping]({{https://blbadger.github.io}}/3_body_problem/cuda_abbreviated.png)

### Data type optimization

Calculations performed in 64-bit double precisoin floating point format are in the case of the three body problem not optimally efficient.  This is because double precision floating point number contain 11 bits for the 













