##Masked Mixers III: Custom CUDA kernels

### Introduction



### Getting Started

Developing your own CUDA kernels usually requires some form of the Nvidia CUDA Compiler (NVCC) driver. 


Containerization of this process allows you (the developer) to spin up a self-contained virtual environment that can match your desired CUDA runtime, driver and host library (in this case `Torch`) versions. But it also brings a level of complexity: most container runtimes like Docker are unable to interact with specialized hardware like GPUs without some help in the form of driver toolkits. 

After some experimentation, I was able to 

pc_nvidia-smi

```
docker pull blbadger/cuda120_pytorch240
```

For my 4x V100 (Volta, compute capability 7.0) this container unfortunately fails to compile our test custom CUDA extensions to Pytorch, so I put together another container that is by matching the CUDA minor version

![server_nvidiasmi]

```
docker pull blbadger/cuda122_pytorch240
```

