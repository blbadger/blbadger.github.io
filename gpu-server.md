## Deep Learning Server

When one thinks of state-of-the-art deep learning models, one might tend to think of enormous data centers with thousands of GPUs training models with billions or trillions of parameters. While large computing clusters are indeed necessary for training a variety of models, they are not at all necessary to do interesting work in the field.

On this page, the author details the installation of a GPU server node for half the price of a Mac pro.

### Background

For the past couple of years, I have been using the following system for experimentation and model development (as well as three body simulation and polynomial root fractal generation and other assorted projects): 

![Desktop]({{https://blbadger.github.io}}/server_setup/desktop.jpg)

which is an i7-12700F with 48GB RAM, an RTX 3060 with 12 GB vRAM and (more recently added) a GTX 1060 with 6GB vRAM all on a Gigabytre B660 mobo. This is a very capable system for smaller experiments, and I have used it for the majority of the deep learning work I have written about in preprints or on the blog.

But this system is certainly not ideal for training models or other experimentation. This is because the GPU compute one can bring to bear (the 3060) has relatively low memory bandwidth (360 GBps) and CUDA core count (3584) such that it is not particularly fast at training the models that it can fit in the 12 GB vRAM, even accounting for the FP16 support and asynchronous memory copy features that are present in Nvidia's Ampere architecture. The 1060 was really only added for small-scale experiment work and software development purposes, as it is significantly slower (around two thirds the speed to be precise) than the 3060 and therefore would slow down training significantly were it to be used in parallel. This is a notable difference from applications such as crypto mining, where the cluster runs at a sum speed of all individual GPUs. For deep learning the cluster will typically increase in speed with more parallelization, but will also be bottlenecked by slower GPUs if there is a large enough difference in their compute (especially for training).

I typically perform training runs of no more than a few days, on models that are under 1 billion parameters (for foundational models) due to these limitations. To test methods and ideas that appear promising on this relatively modest amount of compute, I wanted to try upgrading my system. The most straightforward way to do this (with some semblence of affordability) would be to buy some used RTX 3090s and use PCIE extenders and a beefier power source or two for my current system. I came very close to choosing this route, but decided against it for a number of reasons: firstly because the B660 mobo has 5 PCIE slots but four of those are 1x PCIE 3.0, meaning 1 GBps data transfer for all GPUs except one. This is not really a problem for inference, but will slow down the gradient synchronization step for very large models during distributed training (especially for 4 or more GPUs). Secondly, 3090s run quite hot (up to 600 watts per gpu) and my PC resides in a small room, and thirdly because mixed precision training (where weights are stored in FP16 and gradients and norms in FP32) is not optimal on these GPUs due to their lack of FP16 acceleration.

It might be wondered: why not just use Google Colab? One may pay to access a V100 or A100 and train models there. Unfortunately Colab has restricted access patterns that are (I am fairly certain) designed to prevent any large-scale training of models in lieu of more scaled back experimentation. This results in runtimes that have insufficient storage for many larger datasets, and very slow data transfer to and from Google Drive, and frequent inability to access more powerful A100 GPUs. Colab tends to lend itself to one-off experiments or small scale training but becomes difficult to work with otherwise.

What about using another cloud GPU provider like Paperspace or Runpod? Depending on your compute budget, these might indeed provide the most efficient way to accomplish your task. I wanted to avoid this route partly because I use cloud GPUs for work and would not not have anywhere near the same cloud compute budget for my own projects, which is kind of a bummer when there is a direct comparison to be made. But mostly I wanted to go the hardware route because it is much more fun and rewarding to build a system and tailor it to exactly what you are looking for. It turns out that for my use it is much cheaper, too.

### Motivation

My interest in building a deep learning server came after coming to the estimate that it would be significantly more expensive to buy three 3090s (without upgrading anything else at all) than it would be to build an entire v100 server that would be more capable for DL training by most metrics. I had the same general experience years ago with [high voltage engineering projects](https://blbadger.github.io/#high-voltage): obsolete (by industry standards) industrial equipment is often the cheapest way of accomplishing engineering tasks provided that one has the know-how to work with the equipment out of its intended use niche. On this page I will show you how to do exactly that in the context of a deep learning server.

The model I chose is a Gigabyte T180-G20, which is a 4x V100 server node built to Facebook's Open Compute Project standard. These are very similar to the more common T181-G20 except that they support Intel Xeon E5 2600 v3 and v4 generation CPUs, whereas the 181 supports Intel Xeon Scalable CPUs (which are effectively the next generation of CPU after the v4 E5s) and has more memory DIMMS (24 versus 16). Because the T180-G20s support older CPUs and potentially less memory than the T181s, they are a good deal cheaper and can be had for under a thousand dollars new. Not bad for a machine that supports up to 750 TFLOPs for FMA (fused multiply-add) matrix operations with up to six V100 GPUs (four sxm2 and two PCIE), 192 GB vRAM with the same configuration, and 1024 GB DDR4 RAM. 

In my configuration only the four SXM2 sockets are used for V100s, with 500 TFLOPs at 64GB vRAM as a starting configuration. These SXM2 sockets are interconnected via 300 GBps NVlink, making these four GPUs behave for all purposes as one large GPU. I use the 16GB rather than the 32GB V100 as they are nearly a factor of 10 cheaper. 

The main exceptions to this are the T181 and T180, and this is because they 

### Power Supply

![psu]({{https://blbadger.github.io}}/server_setup/dell_psu.jpg)

![psu]({{https://blbadger.github.io}}/server_setup/psu_test.jpg)

### Hardware Installation














