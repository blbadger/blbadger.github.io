## Masked Mixers III: Custom CUDA kernels

### Introduction



### Getting Started

Developing your own CUDA kernels usually requires some form of the Nvidia CUDA Compiler (NVCC) driver. When you are developing a kernel for stand-alone use, you can install the Nvidia C t


Containerization of this process allows you (the developer) to spin up a self-contained virtual environment that can match your desired CUDA runtime, driver and host library (in this case `Torch`) versions. But it also brings a level of complexity: most container runtimes like Docker are unable to interact with specialized hardware like GPUs without some help in the form of driver toolkits. 

To deal with this, Nvidia helpfully supplies you with repositories for the CUDA toolkit (including `nvcc`) on their developer website, where you can choose the toolkit version, your operating system, and CPU architecture and find the matching repositories. For example, for CUDA toolkit 12.0 with Ubuntu 22.04 and an x86-64 CPU, [this link](https://developer.nvidia.com/cuda-12-0-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04) gives you the following bash commands

```sh
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
$ sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
$ wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb
$ sudo dpkg -i cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb
$ sudo cp /var/cuda-repo-ubuntu2204-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
$ sudo apt-get update
$ sudo apt-get -y install cuda
```

which installs the toolkit. 

Which toolkit should one use?

![autoencoder architecture](/deep-learning/pc_nvidiasmi.png)

```
docker pull blbadger/cuda120_pytorch240
```

For my 4x V100 (Volta, compute capability 7.0) this container unfortunately fails to compile our test custom CUDA extensions to Pytorch, so I put together another container that is by matching the CUDA minor version, which as you can see from the following image is 12.2. The minor version should not matter for functionality, but for this server I found that the test CUDA extension was non-compilable if it did not match the Nvidia CUDA driver version (see below). This is likely a CUDA implementation bug (on the V100), as minor versions should always be compatible with each other but here are not.

![autoencoder architecture](/deep-learning/server_nvidiasmi.png)

```
blbadger/cuda122_pytorch240
```

At this point, your GPUs will still be invisible to the docker container's runtime: running a container with Nvidia toolkits installed yields the following:

```
WARNING: The NVIDIA Driver was not detected. GPU functionality will not be available.
  Use the NVIDIA Container Toolkit to start this container with GPU support; see
  https://docs.nvidia.com/datacenter/cloud-native/
```

This is a most helpful error message, as following the link leads us (indirectly) to the Nvidia toolkit [documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), which explaines that you must modify the docker daemon file (`/etc/docker/daemon.json`) to allow the Nvidia Container Runtime, which can be done via

```sh
$ sudo nvidia-ctx runtime configure --runtime=docker
$ sudo systemctl restart docker
```

Now you can run a docker container and access your GPUs! You have to include the `--gpus` flag in your run command or the devices will remain invisible to the container. In the following command, we run the `blbadger/cuda120_pytorch240:updated` docker container interactively with all GPUs visible (use `--gpus 0` for device 0 only, etc.) and enter bash upon initialization.

```sh
$ docker run -i -t --gpus all blbadger/cuda120_pytorch240:updated /bin/bash
```



