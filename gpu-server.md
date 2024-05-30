## Deep Learning Server

When one thinks of state-of-the-art deep learning models, one might tend to think of enormous data centers with thousands of GPUs training models with billions or trillions of parameters. While large computing clusters are indeed necessary for training a variety of models, they are not at all necessary to do interesting work in the field.

On this page, I will show you how to build a high-performance compute server node with four V100 GPUs for around the same cost as a single RTX 4090, but with around triple the GPU memory and compute (assuming 16/32 bit mixed precision training).

### Background

For the past couple of years, I have been using the following rig for experimentation and model development (as well as three body simulation and polynomial root fractal generation and other assorted projects presented on my [blog](https://blbadger.github.io)): 

![Desktop]({{https://blbadger.github.io}}/server_setup/desktop.jpg)

which consists of i7-12700F with 48GB RAM, an RTX 3060 with 12 GB vRAM and (more recently added) a GTX 1060 with 6GB vRAM on a Gigabytre B660m motherboard. This is a very capable computer for smaller experiments, and I have used it for the majority of the deep learning work I have written about in preprints or on the blog.

But this system is certainly not ideal for training larger models or more extensive experimentation. This is because the GPU compute one can bring to bear (the 3060) has relatively low memory bandwidth (360 GBps) and CUDA core count (3584) such that it is not particularly fast at training the models that it can fit in the 12 GB vRAM, even accounting for the FP16 support and asynchronous memory copy features that are present in Nvidia's Ampere architecture. The 1060 was really only added for small-scale experiment work and software development purposes, as it is significantly slower (around two thirds the speed to be precise) than the 3060 and therefore would slow down training significantly were it to be used in parallel. This is a notable difference from other applications such as crypto mining, where the cluster runs at a sum speed of all individual GPUs. For deep learning the cluster will typically increase in speed with more parallelization, but will also be bottlenecked by slower GPUs if there is a large enough difference in their compute (especially for training). I typically perform training runs of no more than a few days, on models that are under 1 billion parameters (for foundational models) due to these limitations. To test methods and ideas that appear promising on this relatively modest amount of compute, I wanted to try upgrading my system. 

The most straightforward way to make a significant upgrade to this rig (with some semblance of affordability) would be to buy some used RTX 3090s and use PCIE extenders and a beefier power source or two for my current system. I came very close to choosing this route, but decided against it for a number of reasons: firstly because the B660 mobo has 5 PCIE slots but four of those are 1x PCIE 3.0, meaning 1 GBps data transfer for all GPUs except one. This is not really a problem for inference, but will slow down the gradient synchronization step for very large models during distributed training (especially for 4 or more GPUs). Secondly, 3090s run quite hot (up to 600 watts per gpu) and my PC resides in a small room, and thirdly because mixed precision training (where weights are stored in FP16 and gradients and norms in FP32) is not optimal on these GPUs due to their lack of full FP16 acceleration.

When deciding on the right GPU depends on many factors (inference or training workloads? large transformers or smaller cNNs?), a good resource to check when deciding on a GPU for deep learning is on Tim Dettmer's [blog](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/). A very handy plot of the theoretical mixed precision training compute for each type of modern GPU is presented, but with some more research I noticed something curious: by far the best-value GPU by current prices (ie on ebay) is the 16 GB V100 SXM2. This is mostly because the SXM2 socket is relatively rare, but we will find a good system for this hardware without too much problem.

Because of this I decided to go with the V100, which is overall a similarly powerful GPU to the 3090 with nearly identical memory bandwidth (897 vs 936 GB/s), albeit double the Tensor core count with half the CUDA cores.  The main functional difference is that the V100 is a data center card rather than a consumer one and is more suitable for extended workloads, and does not suffer from the memory overheating problems the 3090 is known for. It does have an older architecture (Volta) than the 3090 (Ampere) and therefore cannot take advantage of a number of improvements in Ampere such as asynchronous memory copy, but it does has other advantages (full FP16 performance and more memory in the 32 GB version). Using the SXM2 socket happily allows for much faster data transfer between GPUs than the more expensive PCIE version, so it is a no-brainer for training.

It might be wondered: why not just use Google Colab? One may pay to access a V100 or A100 and train models there. Unfortunately Colab has restricted access patterns that are (I am fairly certain) designed to prevent any large-scale training of models in lieu of more scaled back experimentation. This results in runtimes that have insufficient storage for many larger datasets, and very slow data transfer to and from Google Drive, and frequent inability to access more powerful A100 GPUs. Colab tends to lend itself to one-off experiments or small scale training but becomes difficult to work with otherwise.

What about using another cloud GPU provider like Paperspace or Runpod? Depending on your compute budget, these might indeed provide the most efficient way to accomplish your task. I wanted to avoid this route partly because I use cloud GPUs for work and would not not have anywhere near the same cloud compute budget for my own projects, which is kind of a bummer when there is a direct comparison to be made. But mostly I wanted to go the hardware route because it is much more fun and rewarding to build a system and tailor it to exactly what you are looking for. It turns out that for my use it is much cheaper, too.

### Motivation

A similar general principle for procuring parts is found here that also was observed for [high voltage engineering projects](https://blbadger.github.io/#high-voltage): obsolete (by industry standards) industrial equipment is often the cheapest way of accomplishing engineering tasks provided that one has the know-how to work with the equipment out of its intended use niche. On this page I will show you how to do exactly that in the context of a deep learning server.

Most deep learning servers with decent GPUs cost many thousands of dollars to buy used, much less new. The main exceptions to this are the Gigabyte T181-G20 and T180-G20, and this is because these servers are built to fit in and be powered by Open Compute Project 1OU racks. These racks are extremely rare and expensive, making even new T181s and T180s relatively indexpensive. Happily, however, these servers run perfectly well outside the OCP rack if supplied with external power from a sufficiently powerful source (12V with at least 80 amps to each of three power sockets). How this may be done will be described later.

The model I chose is a Gigabyte T180-G20, which is very similar to the T181-G20 except that it supports Intel Xeon E5 2600 v3 and v4 generation CPUs, whereas the T-181 supports Intel Xeon Scalable CPUs (which are effectively the next generation of Intel server CPU after the v4 E5s) and has more memory DIMMS (24 versus 16). For more information on the difference between these servers as well as the other OCP rack entrant from Gigabyte, see [this documentation](https://www.gigabyte.com/FileUpload/TW/MicroSite/354/dl/RACKLUTION-OP_Brochure.pdf) from Gigabyte.

Because the T180-G20 supports older CPUs and less maximum memory than the T181s, they are a good deal cheaper and can be had for under a thousand dollars new. Not bad for a machine that supports up to 750 TFLOPs for FMA (fused multiply-add) matrix operations with up to six V100 GPUs (four sxm2 and two PCIE), 192 GB vRAM with the same configuration, and 1.024 TB DDR4 RAM. In my initial configuration only the four SXM2 sockets are occupied, each with 16GB V100s with 500 TFLOPs at 64GB vRAM. These SXM2 sockets are interconnected via 300 GBps NVlink, making these four GPUs behave for all purposes as one large GPU performance-wise. I chose the 16GB rather than the 32GB V100 as they are nearly a factor of 10 cheaper at present.

### Hardware Installation

Upon receiving the T180-G20 and opening the box, I was met with the following sight:

![heatsinks]({{https://blbadger.github.io}}/server_setup/boxed_heatsinks.jpg)

These are the heatsinks for the GPUs (top four) and CPUs (bottom two). These heatsinks contain all the screws necessary pre-installed as well as thermal paste pre-applied, which is a very nice touch but is probably standard in the high performance compute industry. After removing the heatsinks, we find the server itself. In the following picture, the server is oriented where the power sockets are on the left and the I/O ports (and air inlets) are to the right. Note the hardware installation instructions happily present on the top of the case, and the opened HDD/SSD drive tray in the bottom right.

![server]({{https://blbadger.github.io}}/server_setup/server_lid.jpg)

The server even comes with those little storage drive screws which fixes the drive in place as the tray is inserted.

![server]({{https://blbadger.github.io}}/server_setup/hard_drive_screws.jpg)

This server takes two CPUs which are interconnected and act for most purposes as a single unit. I thought that the Intel Xeon E5 2680 V4 would be a good balance between TDP (120 Watts each) and power (3.3 GHz turbo, with 28 threads, 40 PCIE lanes, and 35 MB caches each). It is remarkable that a CPU of these attributes can be bought for under 20 dollars: to buy a consumer CPU with anywhere near the thread count or PCIE lanes one would have to pay perhaps a hundred times that amount. This CPU has far more pins than the i7 12700K, and the lid is somewhat reminiscent of an AMD Ryzen. It tends to make removing thermal paste messy, as you can see,

![server]({{https://blbadger.github.io}}/server_setup/xeon_pins.jpg)

and here it is ready to be clipped in.

![server]({{https://blbadger.github.io}}/server_setup/cpu_seat.jpg)

In terms of core count two CPUs are extremely overkill for most of the types of workloads I am expecting to run on this machine, that is, CUDA kernals for deep learning and dynamical systems. As we will see later, only one CPU is necessary for excellent performance (and really only a quarter of the cores of one CPU are required). A more powerful CPU does come in handy when one is attempting to perform linear algebraic operations outside the set of matrix multiply, add, convolve, etc: for example, finding the singular value decomposition of a matrix is typically a CPU-intensive process, and there it helps to have all the threads you can get. I will be performing this kind of operation non-infrequently, hence the more powerful CPUs. In terms of functional performance, the 2x Xeon 2680s are about as fast for tokenization as my (much newer) i7-12700K but support nearly 10x the memory and PCIE lanes.

The CPU heatsinks are interesting: only half the base plate contains fins, and the whole heatsink is a copper alloy. In the following image you can also see one memory stick installed: this is a 16GB RDIMM RAM module for testing (more were added later). As with most servers, only RDIMM or LRDIMM modules may be used.

![server]({{https://blbadger.github.io}}/server_setup/cpu_heatsink.jpg)

With CPU heatsinks installed, I installed one GPU for testing purposes. In the image below, the four SXM2 sockets are on the left, CPUs on the right, and PCI-E sockets are in the center right. Note the connection to the SXM2 board from one PCI-E socket, leaving the other two free. This connection from CPU to GPUs is limited to 16 GBps, but data transfer between GPUs on the SXM2 board is a juicy 300 GBps.

![server]({{https://blbadger.github.io}}/server_setup/server_internals.jpg)

The V100 arrived nice and clean with a mirror finish on the lid. The large grey 'TR21...' modules are voltage regulators, and if you have seen an SXM2 P100 this will look very familiar except for the GV100 chip. 

![server]({{https://blbadger.github.io}}/server_setup/gpu_lid.jpg)

Below is the SXM2 socket with protecting plastic and acrylic fan ducting removed. Two screws have to be removed before installing each V100.

![server]({{https://blbadger.github.io}}/server_setup/gpu_socket.jpg)

The hardware installation guide warns you that there is a very fine tolerance window for the screws that fasten the GPU to SXM2 board: less than 5%! This is because there are tiny springs used to modulate torque. It is recommended to use a precision torque screwdriver for installation, but I winged it with a small-bore screwdriver and lots of patience. To be honest, I would probably just get a precision screwdriver if I were to do this again: I had to go back and re-tighten both heatsink and GPU-board connections multiple times to eliminate various gremlins (a too-warm GPU, GPU that was not recognized at all, strange memory hangs resulting in process kills etc). To be frank, the SXM2 connection is not nearly as robust as a modern CPU connection, but this is a small price to pay for huge bandwidth I suppose.

![server]({{https://blbadger.github.io}}/server_setup/gpu_presink.jpg)

The GPU heatsink comes pre-loaded with thermal paste, nice!

![server]({{https://blbadger.github.io}}/server_setup/gpu_heatshink.jpg)

The heatsink has a nice thick base plate and large fins, and is around six times the size of the CPU heatsink. 

![server]({{https://blbadger.github.io}}/server_setup/gpu_heatsink.jpg)

There do not appear to be any heat pipes in the GPU's heatsink, just a hefty base plate (the center of which is copper) and closely spaced fins.

![server]({{https://blbadger.github.io}}/server_setup/through_gpu.jpg)

The heatsink also has a small tolerance window, but with springs it is not quite as small. In the image on the right (below), air moves from right to left.

![server]({{https://blbadger.github.io}}/server_setup/gpu_heatsink_install.png)

Here is a side view of the GPU and heatsink after installation. Interestingly the voltage regulators do not contact the heatsink.

![server]({{https://blbadger.github.io}}/server_setup/gpu_side.jpg)

Replacing the acrylic duct, we have the first GPU installed.

![server]({{https://blbadger.github.io}}/server_setup/installed_gpu.jpg)

### Power Supply Units

Now we get to a tricky part: powering a 1OU OCP server outside its native habitat (an OCP rack) using power supply units (PSUs) designed for different servers. Whilst most of the safety features of PSUs are retained when doing this (over-draw leads to current trip and shutdown etc.) this is obviously not the intended use of these components and therefore the manufacturers cannot be expected to have planned for it. If you are planning on assembling your own OCP server, proceed at your own risk.

With a total TDP of around 1500 watts for the chips alone, perhaps the simplest power supply would be a 2000W dell server PSU. Unfortunately these require 240V AC inputs, and I only have 120V outlets. Happily however other dell server PSUs are designed to be run in parallel as they would be in their native environment, so instead we can just use two 1100W PSUs in parallel (with the current sharing pin connected to avoid burning out one while the other idles). 1100W dell PSUs are very inexpensive on ebay, so I got a couple and started breaking them out, ie making them run outside their intended environment. I started with dell z1100p PSUs, not to be confused with l1100e supplies that have very different pinouts.

![psu]({{https://blbadger.github.io}}/server_setup/dell_psu.jpg)

One can buy breakout boards for this purpose, but I thought it would be more fun to solder the connections myself. This turned out to be more difficult than I had anticipated, and it turns out that you have to first draw one pin to a GND, supply power and wait a few seconds, and then connect two 'switch' pins together to complete the power on. One of the strangest sequences I have seen for a power supply, but at least it works.

![psu]({{https://blbadger.github.io}}/server_setup/psu_test.jpg)

Power must be supplied to each of the sockets on the right of the server in the photo below (actually one can also bypass the sockets and install a cable into the cages next to the sockets, but I thought this would be more of a pain to do). Each socket has a +12V (left) and GND (right) connection, and the server expects 80 amps per socket.

To get the power from the PSU to sockets I used a combination of 4 AWG battery cable and 1/4" solid copper bus bar used to insert into the sockets, and for collecting the inputs for the +12V and GND. The 4 AWG cable turned out to be overkill, and I would choose 6 or 8 AWG if doing this over.

![server]({{https://blbadger.github.io}}/server_setup/server_prepower.jpg)

Sawing the bus bar into chunks and drilling for connections allows for one bus bar in each power socket.

![server]({{https://blbadger.github.io}}/server_setup/bus_terminals.jpg)

Connecting things together with the appropriate hap-hazardness that indicates a test, we have a successful power connection.

![server]({{https://blbadger.github.io}}/server_setup/test_psu.jpg)

While performing these tests, I noticed that this PSU tended to modulate its fans in response to current draw (which is good) but that it tended to be rather warm when the system itself was powered down (bad, indicates parasitic current draw). Because of this (and because I accidentally stripped a pin during a de-solder process of one of the PSUs) I switched my original plan to instead use a similar PSU but with breakout boards. 

The new PSUs are two Dell l1100e-s1 modules with adjustable breakout boards from ebay. There is some voltage drop from the breakout board output to the server power socket, but both PSUs are recruited during heavy workloads even without further coordination. This can be seen during experiments where the wattage pulled is greater than a single PSU's rating: doing so results in a reset of the PSU and system crash if one is used, but with two there is no such failure.

![server]({{https://blbadger.github.io}}/server_setup/final_psu.jpg)

I also connected the current share pins (which is pin S7) of the two PSUs in an effort to get them to coordinate better under high load. This connection can be with very thin wire as it carries virtually no load (<500mW), so I used insulated AWG 20 wire (black in the picture below). I alwo used proper copper ring lugs to make the high-current connections to and from the sockets.

![server]({{https://blbadger.github.io}}/server_setup/gpu_cshare.jpg)

This effort was moderately successful, but there is still a sizeable voltage drop under high load (all four GPUs at ~300W + 200W CPUs + 200W other = ~1600W) which can lead to GPU under-volting and GPU bus drop if this amount of current is drawn for extended periods of time.

![server]({{https://blbadger.github.io}}/server_setup/full_psus.jpg)

Thus the dual l1100e-s1 PSUs with breakout boards and current share pins connected certainly provides more current than only one PSU would, but not as much as the two should under full load (2100 Watts at 120V input). In practice this means that the GPUs need to be limited to 250W each for training runs (pulling around 1400W total), which results in minimal performance degradation (<5%). Due to the power and heat and noise reduction for a slightly larger performance degradation, I tend to limit the power even more than this and run at 200W per GPU. But if one were to want to train on full clock speed and power for extended periods of time, I would recommend getting a single 2KW server PSU and wiring a 240V wall socket if necessary, as this route is by far the simplest and maintenance-free route.

### Test

The I/O ports are suprisingly comprehensive for a server: one VGA, three RJ45s, and two USB ports. 

![server]({{https://blbadger.github.io}}/server_setup/server_io.png)

This makes it easy to connect a monitor, keyboard, ethernet connection, and bootable USB (to install ubuntu-server). There is no reason to connect a mouse as ubuntu-server has no GUI be default (although it is easy to install one). As I am going to `ssh` into this server for almost everything, I did not install a Desktop GUI.

![server]({{https://blbadger.github.io}}/server_setup/server_io_connected.jpg)

After powering on, it POSTs! Much of the hardware in this HPC server is managed by Penguin's software, making a linux OS even more fitting.

![server]({{https://blbadger.github.io}}/server_setup/server_post.jpg)

Heading to the BIOS, we find that both CPUs are recognized, the memory is performing at its maximum speed (2400 MHz) and that the server was built in 2017.

![server]({{https://blbadger.github.io}}/server_setup/server_bios.jpg)

After installing ubuntu-server, we can check the internals. All 56 threads are active, and not terribly busy which is great.

![server]({{https://blbadger.github.io}}/server_setup/server_htop.jpg)

After checking that the GPU hardware with `sudo lshw -C display` and finding our V100, installing the proper NVIDIA drivers and rebooting allows interfacing with the GPU. Et voila, our V100 is found and is idling (although in performance mode, interesting).

![server]({{https://blbadger.github.io}}/server_setup/server_nvidia-smi.jpg)

### Installing the rest of the hardware

With the test completed, I went ahead and installed the rest of the GPUs and memory sticks. Unfortunately the rest of the GPUs were not pre-cleaned, so I had to wipe some thermal paste off the chip lids.

![server]({{https://blbadger.github.io}}/server_setup/more_gpus.jpg)

I seated the GPUs first before installing the heatsinks. Note that you should never attempt to run a GPU without a heatsink! It will rapidly overheat, although it may turn off automatically if you are lucky.

![server]({{https://blbadger.github.io}}/server_setup/all_gpus.jpg)

And here is the server with all the GPUs, memory, and heatsinks installed! 

![server]({{https://blbadger.github.io}}/server_setup/full_gpus.jpg)

I had to re-install two of the GPUs a couple times in order to get them to be recognized by the system, and whether this was due to dusty SXM2 pins, incorrectly torqued screws, or just old hardware it is difficult to tell. Happily it is easy to see if a GPU is connected using the `sudo lshw -C display` command

![server]({{https://blbadger.github.io}}/server_setup/full_lshw.jpg)

and the nvidia toolkit finds the GPUs as well, nice! I had installed fairly recent CUDA driver (535.172.04) and API (12.2) versions, and you can see that here. Note that the GPUs are by default in maximum performance mode (P0) even while idling: this is to be typical of SXM-socketed nvidia GPUs, and indeed the SXM2 V100 cannot be set to any other mode. Note too that watts per GPU at idle has roughly doubled from what it was when a single GPU was installed: this is unsurprising, given that the NVLink connections between GPUs cannot completely idle.

![server]({{https://blbadger.github.io}}/server_setup/full_nvidiasmi.jpg)

We have an impressive 500 teraflops of matmult performance for ~650$ worth of GPUs. A little-appreciated fact is that watt-for-watt the V100's performance is similar to the A100, which has a TDP of 400W (30% more than the V100), but the A100 typically perhaps 45% faster for real workloads. 

### Stress Testing with Deep learning training

Now that the power supply had been sorted out (or so I thought) I went ahead and stress tested the server for workloads typical of what I intended this machine to experience in order to make sure that the GPUs and other hardware elements were performing optimally. 

Launching my `distributed_mixer_trainer.py` module via `torch.distributed.launch` for DDP training via

```bash
-m torch.distributed.launch \
  --nproc_per_node=4 \
  --nnodes=1 \
  distributed_mixer_trainer.py 
```

I was in for a surprise: using up to three GPUs (`--nproc_per_node=3`) performed as expected, but adding the fourth caused the GPUs to drop off the bus, or in other words to crash so completely that they cannot be found via `nvidia-smi -q`. The error messages were extremely unhelpful, and essentially conveyed that there was an unknown failure with the GPUs themselves. 

After some reflection I wondered whether this could be due to the power draw being too much for my power supply: even though the supply should be good for 2100W (175 Amps at the terminal), this server is actually designed to be supplied by 200 Amps per terminal and thus is being somewhat underpowered. But if the TDP of the server is 1500W, why would this matter? It turns out that V100s (like A100s and other high-performance datacenter GPUs) are capable of drawing much more current than their maximum rating for short periods of time during boos: I have seen a 16GB V100 SMX2 on this system (rated at 300W) draw nearly 400W. Even this should not be a problem for our server PSUs, but a rapid increase in load (for a PSU not designed for this) might be: in tenths of a second, it is not uncommon to see a V100 go from 150W to 350W. If you multiply this by four, you get nearly 1KW in rapid load increases, which might lead to undervolting. When I tested the power socket voltage during this ramp-up, it indeed did drop to nearly 11V.

A straightforward way to prevent this issue would be to simply limit the amount of current the GPUs may draw. This may be done via `nvidia-smi -pl` followed by an integer corresponding to the number of watts per GPU desired. After enforcing this power limit, we indeed see that the average power does decrease to approximate the limit but that performance is very lightly affected, such that going from 300W -> 200W results in a ~15% performance decline for the V100. This is even less that what was observed for [consumer graphics cards](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/), but I ended up having the same GPU bus drop problem as before under full load. 

Close inspection with `watch -n0.1 nvidia-smi` revealed the cause: the average current was lower but the peak current was far above the specified 200 watts per GPU, and even exceeded 300 watts! Clearly the V100s viewed the power limit as more guidelines rather than actual rules, so I resorted to down-clocking the processor to avoid boost as follows:

```bash
bbadger@servbadge:~/experiments/generative-models/mixer_lm$ sudo nvidia-smi -pm 1
bbadger@servbadge:~/experiments/generative-models/mixer_lm$ sudo nvidia-smi -ac 877,1005
```
where the first command induces persistance and the second specifies the application clock speed in format `<memory_clock,processor_clock`. The exact processor and memory clocks that are allowed are somewhat arbitrary depending on the GPU, and you will have to use `nvidia-smi -q -d SUPPORTED_CLOCKS` to view the clock limits that are supported. Newer GPUs have an `-lgc` flag that can be used to specify a logic versus application clock, but this flag is not present for the V100. 

After applying the clock limit, we have GPUs that are now observing our intended power limit, and training proceeds successfully.

![server]({{https://blbadger.github.io}}/server_setup/training_gpu_power.png)

Note that the GPU in position 2 is running significantly hotter than the others: this is only under load, and was due to an incorrectly torqued heatsink. After tightening, the GPU is more in line with the others and no longer ramps up the fan speed (more on that later).

![server]({{https://blbadger.github.io}}/server_setup/idle_gpu_power.png)

It is remarkable that removing the V100 boost clock speed (1530 MHz) and reducing the base from 1290 MHz to 1005 MHz (along with our earlier power limits) leads to such a small change in performance: enforcing this for a training run with only two GPUs during a test leads to a ~13% decline in training speed.

That said, once these tests were completed I connected the PSU current share pins (see the PSU section for more information) and after doing so the GPUs were able to run at 300W each without problems. I consider the 400W total saved to be worth a 13% performance drop, and continue to run the GPUs as 200W each.

I should mention two unexpected hiccups: firstly, running `watch -n0.1 nvidia-smi` leads to significantly slower training especially when more than one GPU is used: there is a 15% performance drop when all four GPUs are running at full tilt (although only a ~5% drop for one GPU). This is a much more noticeable drop than for consumer grade hardware (RTX 3060 with i7-12700F) and it is unknown why it occurs. A simple test to see if this was caused by the NVLink system would be to remove all GPUs save one and repeat the same training runs with `nvidia-smi` running concurrently, but I will not be doing this as removing and replacing SXM2 modules is a pain.

Secondly, I experienced difficult-to-pin-down training process hangs and freezes which manifested as `nv_queue` processes taking nearly all a CPU's compute followed by interrupt requests (`irq/57-nvidia`) that also froze. Close inspection reveals that these are associated with `nccl` communication problems, which means the GPUs are not communicating properly with each other or the CPU. I was able to solve this problem by simply tightening the screws that affix the GPUs to the SXM2 socket.

As a final note, `python3 -m torch.distributed.launch` is a legacy DDP launcher, and I prefer `torchrun` as it is easier and slightly more performant to work with for this server.

### Performance

Now we can test the performance. Happily it is very good! Depending on the workload, each V100 is between two and four times faster than my RTX 3060. The exact difference depends somewhat on the task at hand, and appears to mostly be the result of the difference in memory between these GPUs: the 3060 uses GDDR6 (fast clock, low bandwidth) and the V100 uses HBM2 (slow clock, large bandwidth). Thus for models with small weight matrices the 3060 is relatively better-suited, but for larger models the V100's HBM2 becomes far superior. In my tests on a 2048-model dimensional [language mixer](https://blbadger.github.io/smaller-lms.html), a mixed precision training run with the 3060 took 880 seconds, whereas a single V100 took 286. This ~3.1x speedup seems typical of medium-sized models for mixed precision training, and aligns with what is expected from theoretical values from Dettmer's [blog post](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/).

Now we get to enjoy the fruits of our SXM2 socket labor as well: because the inter-GPU bandwidth is a whopping 300GB/s, there is virtually no per-GPU performance decrease when parallelizing a workload using distributed data parallel for a medium-small model (~300m parameters) with some current and clock limiting for power reduction: with one GPU a similar training run took 361 seconds, with two 180 seconds, with three 121 seconds, and with all 4 GPUs 92 seconds. This sort of thing is generally not possible NVLink: it is common to see speedups of between 3.5x and 3.6x for four GPUs that are connected by 16x PCIE lanes to the CPU. Here the very high GPU communication bandwidth allows us to hit a speedup of 3.92x for four GPUs! The same speedups are observed even when the GPU application clock is not limited to 1005 MHz, although in each case there is an approximately ~14% reduction in time at the cost of around 500 watts.

To substantiate the claims made earlier that the CPU core number is quite overkill for training deep learning models, observe the single-Xeon 2680 (with 28 threads) CPU utilization for a CPU-intensive task such as fast tokenization,

![server]({{https://blbadger.github.io}}/server_setup/cpu_tokenization.png)

or a similarly intensive task of performing forward- and back-propegation on a medium-sized model.

![server]({{https://blbadger.github.io}}/server_setup/cpu_training.png)

In both cases the majority of thread are heavily utilized. Now for the thread utilization for four-V100 DDP training:

![server]({{https://blbadger.github.io}}/server_setup/all_gpu_training.png)

the cores utilized count is small because the DDP defaults to a single thread per GPU, with a few threads saved for data loading. Increasing the number of threads per GPU in my experience does not result in better performance and indeed often leads to detrimental effects. Because of this, we can estimate that a 28-thread CPU could support nearly two dozen GPUs if the PCIE lane count were high enough! The 4x V100 SXM2 board requires two 16x PCIE lanes, so 20x V100s would require 160 PCIE lanes.

That said, having a large number of cores can come in very handy for CPU-bound tasks such as tokenization or more complicated linear algebraic

### Noise

This was one of the things I though hardest about before going the T180/T181 route over a bunch of used 3090s in a PC. The world's best server is useless in a home setting if it has the acoustic properties of a turbojet engine, unless one were to make special accommodations such as walling off the server in concrete. This sort of thing did not appeal to me, and while the server was going to be operating in a basement room and could be noisier than the average PC it could not be overly loud.

The reputation of 1U servers (the more common measurement that is most similar to the T180-G20's 1OU form factor) is that they are simply too loud for home use and that they indeed sound like jet engines. This much was even claimed by George Hotz while talking about the motivations for Tinygrad's Tinybox, but I can confirm that it is a bit of a misunderstanding. The potential for a high-performance compute 1OU server such as the T180 for making noise is indeed very high: when first booting up, for example, all 50 of the fans ramp up to their maximum 25000 RPM and the server sounds alot like a jet engine during takeoff, such that one needs hearing protection to work with it in a small room (testing or setting up ssh, for example). The fans to modulate after a couple minutes, and the noise becomes much more managable and is what one would expect for a blade server: equivalent to a somewhat noisy Desktop PC, just with a timbre more akin to a jet.

More importantly, even heavy loads on all four V100s does not lead to the fans reaching anywhere near their maximum RPM provided the ambient temperature is near room temp (72* F). I would not want to run a 1OU HPC server like the T180-G20 in a living room, but in a basement or attic or garage it is virtually unnoticeable outside the room itself. If you have ever heard a fan-cooled ebay grow light before, it sounds pretty much identically to that during normal operation.

### Conclusion

To conclude, you can build a 4x SXM2 V100 server that is very good at all sorts of things for around 950 (server) + 160*4 (gpus) + 30 (cpus) + 80 (mem) + 50 (ssd) + 25 (PSUs) = 1775 USD if you are willing to hack a power supply and be patient and somewhat resourceful. I reckon that this system is overkill for most types of machine learning model inference, but for training it is great and it is very good at performing simulations too. If you have more cash you could build three of these with 32GB V100s and connect via mellanox fabrics, which would allow for training of models up to around 500 billion parameters.


