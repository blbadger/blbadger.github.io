## Image Classification with Deep Learning

### Introduction

In scientific research it is often important to be able to categorize samples accurately. With experience, the human eye is in many instances able to accurately discern and categorize many important features from microsopic observation. 

Another way to accomplish this task is to employ a computational method.  One could imagine attempting to program a machine to recognize the same features that an expert individual would take note of in order to replicate that individual's performance.  Historically this approach was taken in efforts resulting in what were called 'expert systems', but now such approaches are generally no longer attempted.  This is for a couple reasons: firstly because they are extremely tedious and time-consuming to implement, and secondly because once implemented these methods generally fail to be very accurate. The underlying cause for both of these points of failure is that it is actually very difficult to explain complicated tasks like image recognition precisely yet generally enough to be accurate.  

The failure of attempts to directly replicate complex tasks programatically has led to increased interest in the use of probability theory to make estimations rather than absolute statements, resulting in the expansion of a field of statistical learning.  Particularly difficult tasks like image classification often require particularly detailed computational methods for best accuracy, and this computationally-focused statistical learning is called machine learning.

### Implementing a neural network to classify images

The current state-of-the-art method for classifying images via machine learning is achieved with neural networks.  These are programs that use nodes called artificial 'neurons' that have associated weights and biases that are adjusted in accordance to some loss function in order to 'learn' during training.  These neuron are arranged in sequential layers, each representing the input in a potentially more abstract manner before the output layer is used for classification.  This sequential representation of the input in order to accomplish a machine learning task is the core idea behind the field of deep learning, encompasses artifical neural networks along with other models such as Boltzmann machines.

A hands-on introduction to the theory and utility of neural networks for image classification is found in Nielsen's [book](http://neuralnetworksanddeeplearning.com/), and the core algorithms of stochastic gradient descent and backpropegation that are used to train neural nets on this page are explained there.  For a deeper and more comprehensive study of this and related topics, perhaps the best resource is the [classic text](https://www.deeplearningbook.org/) by Goodfellow, Bengio, and Courville.  This page will continue with the assumption of some familiarity with the fundamental ideas behind deep learning.

Neural networks are by no means simple constructions, but are not so prohibitively complicated that they cannot be constructed from scratch such that each operation to every individual neuron is clearly defined (see [this repo](https://github.com/blbadger/neural-network/blob/master/connected/fcnetwork_scratch.py) for an example).  But this approach is relatively slow: computing a tensor product element-wise in high-level programming languages (C, C++, Python etc) is much less efficient than computation with low-level array optimized libraries like BLAS and LAPACK.  In python, `numpy` is perhaps the most well-known and powerful library for general matrix manipulation, and this library can be used to simplify and speed up neural network implementations, as seen [here](https://github.com/blbadger/neural-network/blob/master/connected/fcnetwork.py). 

As effective as numpy is, it is not quite ideal for speedy computations with very large arrays, specifically because it does not optimize memory allocation and cannot make use of a GPU for many simultaneous calculations. To remedy this situation, there are a number of libraries were written  for the purpose of rapid computations with the tensors, with Tensorflow and PyTorch being the most popular.  Most of this page references code employing Tensorflow (with a Keras front end) and was inspired by the [Keras introduction](https://www.tensorflow.org/tutorials/keras/classification) and [load image](https://www.tensorflow.org/tutorials/load_data/images?hl=TR) tutorials.  

The first thing to do is to prepare the training and test datasets.  Neural networks work best with large training sets composed of thousands of images, so we split up images of hundreds of cells into many smaller images of one or a few cells.  We can also performed a series of rotations on these images:  rotations and translation of images are commonly used to increase the size of the dataset of interest.  These are both types of data augmentation, which is when some dataset is expanded by a defined procedure.  Data augmentation should maintain all invariants in the original dataset in order to be representative of the information found there, and in this case we can perform arbitrary translations and rotations.  But for other image sets, this is not true: for example, rotating a 6 by 180 degrees yields a 9 meaning that arbitrary rotation does not maintain the invariants of images of digits.

Code that implements this procedure is found [here](https://github.com/blbadger/nnetworks/blob/master/NN_prep_snippets.py). Similar image preparation methods are also contained in Tensorflow's preprocessing module.

Now let's design the neural network architecture.  We shall be using established libraries for this set, so reading the documentation for [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/api_docs) may be useful.  First we write a docstring for our program stating its purpose and output, and import the relevant libraries.


```python
# Standard library
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import pathlib

# Third-party libraries
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 

```

Now we can assign the directories storing the training and test datasets assembled above to variables that will be called later.  Here `data_dir` is our training dataset and `data_dir2` and `data_dir3` are test datasets, which link to folders I have on my Desktop.

```python
def data_import():
	"""
	Import images and convert to tensorflow.keras.preprocessing.image.ImageDataGenerator
	object

	Args:
		None

	Returns:
		train_data_gen1: ImageDataGenerator object of image for training the neural net
		test_data_gen1: ImageDataGenerator object of test images
		test_data_gen2: ImageDataGenerator object of test images
	"""

	data_dir = pathlib.Path('/home/bbadger/Desktop/neural_network_images',  fname='Combined')
	data_dir2 = pathlib.Path('/home/bbadger/Desktop/neural_network_images2', fname='Combined')
	data_dir3 = pathlib.Path('/home/bbadger/Desktop/neural_network_images3', fname='Combined')
```

Now comes a troubleshooting step.  In both Ubuntu and MacOSX, `pathlib.Path` sometimes recognizes folders or files ending in `._.DS_Store` or a variation on this pattern.  These folders are empty and appear to be an artefact of using `pathlib`, as they are not present if the directory is listed in a terminal, and these interfere with proper classification by the neural network.  To see if there are any of these phantom files,

```python
	image_count = len(list(data_dir.glob('*/*.png')))

	CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') 
				if item.name not in ['._.DS_Store', '._DS_Store', '.DS_Store']])

	print (CLASS_NAMES)
	print (image_count)
```
prints the number of files and the names of the folders in the `data_dir` directory of choice.  If the number of files does not match what is observed by listing in terminal, or if there are any folders or files with the `._.DS_Store` ending then one strategy is to simply copy all files of the relevant directory into a new folder and check again.

Now the images can be rescaled (if they haven't been already), as all images will need to be the same size. Here all images are scaled to a heightxwidth of 256x256 pixels.  Then a batch size is specified, which determines the number of images seen for each training epoch.

```python
	### Rescale image bit depth to 8 (if image is 12 or 16 bits) and resize images to 256x256, if necessary
	image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
	IMG_HEIGHT, IMG_WIDTH = 256, 256

	### Determine a batch size, ie the number of image per training epoch
	BATCH_SIZE = 400

```
From the training dataset images located in `data_dir`, a training dataset `train_data_1` is made by calling `image_generator` on `data_dir`.  The batch size specified above is entered, as well as the `target_size` and `classes` and `subset` kwargs.  If shuffling is to be performed between epochs, `shuffle` is set to `True`.  

`image_generator` also has kwargs to specify rotations and translations to expand the training dataset, although neither of these functions are used in this example because the dataset has already been expanded via rotations.  The method returns a generator that can be iterated to obtain images of the dataset.

```python
	train_data_gen1 = image_generator.flow_from_directory(
		directory=str(data_dir),
		batch_size=BATCH_SIZE, 
		shuffle=True, 
		target_size=(IMG_HEIGHT,IMG_WIDTH), 
		classes=list(CLASS_NAMES), 
		subset = 'training')
```

The same process is repeated for the other directories, which in this case contain test image datasets.  An image set generation may be checked with a simple function that plots a subset of images. This is particularly useful when expanding the images using translations or rotations or other methods in the `image_generator` class, as one can view the images after modification.

```python
train_data_gen1, test_data_gen1, test_data_gen2 = data_import()
image_batch, label_batch = next(train_data_gen1)
```

Assigning the pair of labels to each iterable in the relevant generators,

```python
(x_train, y_train) = next(train_data_gen1)
(x_test1, y_test1) = next(test_data_gen1)
(x_test2, y_test2) = next(test_data_gen2)
```

Now comes the fun part: assigning a network architecture! The Keras `Sequential` module is a straightforward, if relatively limited, class that allows a sequential series of network architectures to be added into one model. This does not allow for branching or recurrent architectures, in which case the more flexible functional Keras `tensorflow.keras.Model` should be used, and below is an example of the network implemented with this module. After some trial and error, the following architecture was found to be effective for the relatively noisy images here.

For any neural network applied to image data, the input shape must match the image x- and y- dimensions, and the output should be the same as the number of classification options.  In this case, we are performing a binary classification between two cells, so the output layer of the neural network has 2 neurons and the input is specified by `IMG_HEIGHT, IMG_WIDTH` which in this case is defined above as 256x256.  

The first step when using `keras.Model` is to create a class that inherits from `keras.Model`, and it is a good idea to inherit the objects of the parent class using `super().__init__()` as well

```python
class DeepNetwork(Model):

    def __init__(self):
        super(DeepNetwork, self).__init__()
        self.entry_conv = Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1), data_format='channels_last')
        self.conv16 = Conv2D(16, 3, padding='same', activation='relu')
        self.conv32 = Conv2D(32, 3, padding='same', activation='relu')
        self.conv32_2 = Conv2D(32, 3, padding='same', activation='relu')
        self.conv64 = Conv2D(64, 3, padding='same', activation='relu')
        self.conv64_2 = Conv2D(64, 3, padding='same', activation='relu')
        self.max_pooling = MaxPooling2D()

        self.flatten = Flatten()
        self.d1 = Dense(512, activation='relu')
        self.d2 = Dense(10, activation='softmax')
        
```

The way to read the `Conv2D` layer arguments is as follows: `Conv2D(16, 3, padding='same', activation='relu')` signifies 16 convolutional (aka filter) layers with a kernal size of 3, padded such that the x- and y-dimensions of each convolutional layer do not decrease, using ReLU (rectified linear units) as the neuronal activation function.  The stride length is by default 1 unit in both x- and y-directions.

Now the class method `call` may be defined.  This is a special method for classes that inherit from `Module` and is called every time an input is fed into the network and specifies way the layers initialized by `__init__(self)` are stitched together to make the whole network.

```python
    def call(self, model_input):
        out = self.entry_conv(model_input)
        for _ in range(2):
            out = self.conv16(out)
            out = self.max_pooling(out)
	    
        out2 = self.conv32(out)
        out2 = self.max_pooling(out2)
        for _ in range(2):
            out2 = self.conv32_2(out2)
	    
        out3 = self.max_pooling(out2)
        out3 = self.conv64(out3)
        for _ in range(2):
            out3 = self.conv64_2(out3)
            out3 = self.max_pooling(out3)
	    
        output = self.flatten(out3)
        output = self.d1(output)
        final_output = self.d2(output)
	
        return final_output

```
Now the class `DeepNetwork` can be instantiated as the object `model`

```
model = DeepNetwork()
```

`model` is now a network architecture that may be represented graphically as

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/neural_network.png)

For optimal test classification accuracy at the expense of longer training times, an extra dense layer of 50 neurons was added to the above architecture. This may be accomplished by initializing `self.d2` and adding this to `call()` 

```python
class DeepNetwork(Model):

    def __init__(self):
    ...
    
        self.d1 = Dense(512, activation='relu')
	self.d2 = Dense(50, activation='relu')
        self.d3 = Dense(10, activation='softmax')


	def call(self, model_input):
	...
	    output = self.flatten(out3)
	    output = self.d1(output)
	    output = self.d2(output)
	    final_output = self.d3(output)
	    return final_output
])
```

Now the network can be compiled, trained, and evaluated on the test datasets. `model.summary()` prints the parameters of each layer in our model for clarity.

```python
model.compile(optimizer='Adam', 
	loss = 'categorical_crossentropy', 
	metrics=['accuracy'])

model.summary()
model.fit(x_train, y_train, epochs=9, batch_size = 20, verbose=2)
model.evaluate(x_test1, y_test1, verbose=1)
model.evaluate(x_test2, y_test2, verbose=1)

```

A few notes about this architecture: first, the output is a softmax layer and therefore yields a probability distribution for an easy-to-interpret result.  The data labels are one-hot encoded, meaning that the label is denoted by a vector with one-hot tensor, ie instead of labels such as `[3]` we have `[0, 0, 1]`.  In Tensorflow's lexicon, categorical crossentropy should be used instead of sparse categorical crossentropy because of this.  

### Accurate image classification

We follow a test/train split on this page: the model is trained on ~80% of the sample data and then tested on the remaining 20%, which allows us to estimate the model accuracy using data not exposed to the model during training. For reference, using a blinded approach by eye I classify 93 % of images correctly for certain dataset, which we can call 'Snap29' after the gene name of the protein that is depleted in the cells of half the dataset (termed 'Snap29') along with cells that do not have the protein depleted ('Control').  There is a fairly consistent pattern in these images that differentiates 'Control' from 'Snap29' images: depletion leads to the formation of aggregates of fluorescent protein in 'Snap29' cells.

The network shown above averaged >90 % binary accuracy (over a dozen training runs) for this dataset.  We can see these test images along with their predicted classification ('Control' or 'Snap29'), the confidence the trained network ascribes to each prediction, and the correct or incorrect predictions labelled green or red, respectively.  The confidence of assignment is the same as the activation of the neuron in the final layer representing each possibility.  

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/nn_images_1.png)

Let's see what happens when the network is applied to an image set without a clear difference between the 'Control' and experimental group (this time 'Snf7', named  after the protein depleted from these cells in this instance).  After being blinded to the true classification labels, I correctly classified 71 % of images of this dataset.  This is better than chance (50 % classification accuracy being that this is a balanced binary dataset) and how does the network compare? The average training run results in 62 % classification accuracy.  We can see the results of one particular training run: the network confidently predicts the classification of nearly all images, but despite this confidence it is incorrect for many.

![snf7 test accuracy]({{https://blbadger.github.io}}/neural_networks/nn_images_2.png)

Each time a network is trained, there is variability in how effective the training is even with the same datasets as inputs.  Because of this, it is helpful to observe a network's performance over many training runs (each run starting with a naive network and ending with a trained one)  The statistical language R (with ggplot) can be used to summarize data distributions, and here we compare the test accuracies of networks trained on the two datasets.

```R
# R
library(ggplot2)

data1 = read.csv('~/Desktop/snf7_vs_snap29color_deep.csv')
attach(data1)
fix(data1)
l <- ggplot(data1, aes(comparison, test_accuracy, fill=comparison))
l + geom_boxplot(width=0.4) +
    geom_jitter(alpha=0.5, position=position_jitter(0.1)) +
    theme_bw(base_size=14) + 
    ylim(50, 100) +
    ylab('Test accuracy') +
    xlab('Dataset')
    
```

This yields

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/nn_images_3.png)

### Generalization and training velocity

Why does the network confidently predict incorrect answers for the Snf7 dataset?  Let's see what happens during training.  One way to gain insight into neural network training is to compare the accuracy of training image classification at the end of each epoch.  This can be done in R as follows:

```R
# R
library(ggplot2)
data7 = read.csv('~/Desktop/snf_training_deep.csv')
data6 = read.csv('~/Desktop/snap29_training_deep_2col.csv')

q <- ggplot() +
  # snf7
  geom_smooth(data=data7, aes(epoch, training.accuracy), fill='blue', col='blue', alpha=0.2) +
  geom_jitter(data=data7, aes(epoch, training.accuracy), col='blue', alpha=0.4, position=position_jitter(0.15)) +
  # snap29
  geom_smooth(data=data6, aes(epoch, training.accuracy), fill='red', col='red', alpha=0.2) +
  geom_jitter(data=data6, aes(epoch, training.accuracy), col='red', alpha=0.4, position=position_jitter(0.15))

q + 
  theme_bw(base_size=14) +
  ylab('Training accuracy') +
  xlab('Epoch') +
  ylim(50, 105) + 
  scale_x_continuous(breaks = seq(1, 9, by = 1))
  
```

As backpropegation lowers the cost function, we would expect for the classification accuracy to increase for each epoch trained. For the network trained on the Snap29 dataset, this is indeed the case: the average training accuracy increases in each epoch and reaches ~94 % after the last epoch.  But something very different is observed for the network trained on Snf7 images: a rapid increase to 100 % training accuracy by the third epoch is observed.

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/nn_images_4.png)

The high training accuracy (100%) but low test accuracy (62 %) for the network on Snf7 dataset images is indicative of a phenomenon in statistical learning called overfitting.  Overfitting signifies that a statistical learning procedure is able to accurately fit the training data but fails to generalize previously-unseen test dataset.  Statistical learning models (such as neural networks) with many degrees of freedom are somewhat prone to overfitting because small variations in the training data are able to be captured by the model regardless of whether or not they are important for true classification.  

Overfitting is intuitively similar to memorization: both lead to an assimilation of information previously seen, without this assimilation necessarily helping the prediction power in the future.  One can memorize exactly how letters look in a serif font, but without the ability to generalize then one would be unable to indentify many letters in a sans-serif font.  The goal for statistical learning is to make predictions on hitherto unseen datasets, and thus to avoid memorization which does not guarantee this ability.

Thus the network overfits Snf7 images but is able to generalize for Snap29 images.  This makes intuitive sense from the perspective of manual classification, as there is a relatively clear pattern that one can use to distinguish Snap29 images, but little pattern to identify Snf7 images.  

A slower learning process for generalization is observed compared to that which led to overfitting, but perhaps some other feature could be causing this delay in training accuracy.  The datasets used are noticeably different: Snap29 is in two colors whereas Snf7 is monocolor.  If a deeper network (with the extra layer of 50 dense neurons before the output layer) is trained on a monocolor version of the Snap29 or Snf7 datasets, the test accuracy achieved is nearly identical to what was found before,

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/nn_images_5.png)

And once again Snap29 training accuracy lags behind that of Snf7.

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/nn_images_8.png)

### AlexNet revisited

To see if the faster increase in training accuracy for Snf7 was peculiar to the particular network architecture used, I implemented a model that mimics the groundbreaking architecture now known as [AlexNet](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), and the code to do this may be found [here](https://github.com/blbadger/neural-network/blob/master/AlexNet_sequential.py).  There are a couple differences between my recreation and the original that are worth mentioning: first, the original was split across two machines for training, leading to some parallelization that was not necessary for my training set.  More substantially, AlexNet used a somewhat idiosyncratic normalization method that is related to but distinct from batch normalization, which has been substituted here.  

Using this network, it has previously been seen that overfitting is the result of slower increases in training accuracy relative to general learning (see [this paper](https://arxiv.org/abs/1611.03530) and [another paper](https://dl.acm.org/doi/10.5555/3305381.3305406)).  With the AlexNet clone, once again the training accuracies for Snf7 increased faster than for Snap29 (although test accuracy was poorer for both relative to the deep network above).  This suggests that faster training leading to overfitting in the Snf7 dataset is not peculiar to one particular network architecture and hyperparameter choice.

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/nn_images_10.png)

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/nn_images_11.png)

There are a number of interesting observations from this experiment.  Firstly, the multiple normalization methods employed by AlexNet (relative to no normalization used for our custom architecture) are incapable of preventing severe overfitting for Snf7, or even for Snap29.  Second, with no modification to hyperparameters the AlexNet architecture was able to make significant progress towards classifying Snap29 images, even though this image content is far different from the CIFAR datasets that AlexNet was designed to classify.  The third observation is detailed in the next section.

### How learning occurs

At first glance, the process of changing hundreds of thousands of parameters for a deep learning model seems to be quite mysterious. For a single layer it is perhaps simpler to understand how each component of the input influences the output such that the gradient of each component wrt. a function on the output may be computed, and once that is done the value of that parameter may be changed accordingly.  But with many layers of input representation between the input and output, how does a gradient update to one layer not affect other layers? How accurate is gradient computation using backpropegation for deep learning models?

The answer to the latter question is that it is fairly accurate and not too difficult to compute the gradient itself, but the question of how the successive layers influence each other during gradient update is an outstanding one in the field of deep learning and has no clear answer as yet.

This does not mean that we cannot try to understand how learning occurs regardless. One way we can do this is to observe how the accuracy or model loss function changes over time as a model is fed certain inputs, and this is done in the preceding sections on this page. A more direct question may also be addressed: what does the model learn to 'look at' in the input?

We can define what a model 'looks at' most in the input as the inputs that change the output of the model the most, which is called input attribution.  Attribution may be calculated in a variety of different ways, and here we will use a particularly intuitive method: gradient*input. The gradient of the output with respect to the input, projected onto the input, tells us how each input component changes when the output is moving in the direction of greatest ascent by definition. We simply multiply this gradient by the input itself to find how each input component influences the output.  Navigate over to [this page](https://blbadger.github.io/nn_interpretations.html) for a look at another attribution method and more detail behind how these are motivated.

Thus we are interested in the gradient of the model's output with respect to the input multiplied by the input itself,

$$
\nabla_a O(a; \theta) * a
$$

where $\nabla_a$ is the gradient with respect to the input tensor (in this case a 1x256x256 monocolor image) and $O(a; \theta)$ is the output of our model with parameters $\theta$ and input $a$ and $*$ denotes Hadamard (element-wise) multiplication. This can be implemented as follows:

```python
def gradientxinput(model, input_tensor, output_dim):
	...
	input_tensor.requires_grad = True
	output = model.forward(input_tensor)
	output = output.reshape(1, output_dim).max()

	# backpropegate output gradient to input
	output.backward(retain_graph=True)
	gradientxinput = torch.abs(input_tensor.grad) * input_tensor
	return gradientxinput
```
although note that the figures below also have a max normalization step before returning the gradientxinput tensor.  The `gradientxinput` object returned is a `torch.Tensor()` object and happily may be viewed directly using `matplotlib.pyplot.imshow()`. 

Earlier it was noted that a human may learn to distinguish between a healthy and unhealthy cell by looking for clumps of protein in the images provided.  Does a neural network perform classification the same way?  Applying our input attribution method to one particular example of an unhealthy cell image for a trained model, we have

![gradients]({{https://blbadger.github.io}}/neural_networks/snf7_gradxinput.png)

where the attributions are in purple and the original input image is in grayscale.  Here it may clearly be seen that the clumps of protein overlap with the regions of highest attribution.  It is interesting to note that the same attribution is applied to images of healthy cells: 

![gradients]({{https://blbadger.github.io}}/neural_networks/snf7_gradxinput2.png)

Across many input images, a trained model tend to place the highest attribution on exactly these clumps accross many images ('S' denotes unhealthy and 'C' denotes healthy cells)

![gradients]({{https://blbadger.github.io}}/neural_networks/snf7_gradxinput_grid.png)

As a general rule, therefore, the deep learning models place the most importance on the same features as an expert human when attempting to classify the images. The models effectively learn how to discriminate between classification options by determining how much of a clump of protein exists in the image.

### Learning does not equate to global minimization of a cost function during training

The above experimental results are quite interesting because they suggest that, for deep learning models of quite different architecture, a model's ability to generalize depends not only on the model's parameters $\theta$ and the family of functions that it can approximate given its possible outputs $O_n(\theta; a)$ but also on the choice of input $a$.  Furthermore, the input $a$ is capable of inducing overfitting in a model employing extensive measures to prevent this phenomenon (L2 regularization, batch normalization) but a different input results in minimal overfitting (in the same epoch number) in a model that has none of these measures (nor any other regularizers separate from the model architecture itself).

Why is this the input so important to the behavior of the network?

Neural networks learn by adjusting the weights and biases of the neurons in order to miminize a cost function, which is a continuous and differentiable representation of the accuracy of the output of the network. This is usually done with a variant on stochastic gradient descent, which involves calculating the gradient (direction of largest change) using a randomly chosen subset of images (which is why it is 'stochastic').  This is conceptually similar to rolling a ball representing the network down a surface representing the multidimensional weights and biases of the neurons.  The height of the ball represents the cost function, so the goal is to roll it to the lowest spot during training.  

As we have seen in the case for Snf7 dataset images, this idea is accurate for increases in training accuracy but not not necessarily for general learning: reaching a global minimum for the cost function (100 % accuracy, corresponding to a cost function of less than 1E-4) led to severe overfitting and confident prediction of incorrect classifications.  This phenomenon of overfitting stemming from a global minimum in the cost function during training is not peculiar to this dataset either, as it has also been observed [elsewhere](http://proceedings.mlr.press/v38/choromanska15.pdf).  

If decreasing the neural network cost function is the goal of training, why would an ideal cost function decrease (to a global minimum) not be desirable?  In our analogy of a ball rolling down the hill, something important is left out: the landscape changes after every minibatch (more accurately, after every computation of gradient descent and change to neuronal weights and biases using backpropegation).  Thus as the ball rolls, the landscape changes, and this change depends on where the ball rolls. 

In more precise terms, for any given fixed neural network or similar deep learning approach we can fix the model architecture to include some set of parameters that we change using stochastic gradient descent to minimize some objective function.  The 'height' $h$ of our landscape is the value of the objective function, $J$.

In this idealized scenario, the objective function $F$ is evaluated on an infinite number of input examples, but practically we can only evaluate it on a finite number of training examples.  The output $O$ evaluated on set $a$ of training examples $a$ parametrized by weights and biases $\theta$ is $O(a; \theta)$ such that the loss function $J(O)$ is 

$$
h = J(O(a; \theta))
$$

What is important to recognize is that, in a non-idealized scenario, $h$ can take on many different values for any given model configuration $\theta$ depending on the specific training examples $a$ used to evaluate the objecting function $J$.  Thus the 'landscape' of $h$ changes depending on the order and identity of inputs $j$ fed to the model during training, even for a model of fixed architecture.  This is true for both the frequentist statistical approach in which there is some single optimal value of $\theta$ that minimizes $h$ as well as the Bayesian statistical approach in which the optimal value of $\theta$ is a random variable as it is unkown whereas any estimate of $\theta$ using the data is fixed.

As $j$ is finite, an optimal value (a global minimum) of $h$ can be achieved by using stochastic gradient descent without any form of regularization, provided that the model has the capacity to 'memorize' the inputs $a$. This is called overfitting, and is strenuously avoided in practice because reaching this global minimum nearly always results in a model that performs poorly on data not seen during training.  But it is important to note that this process of overfitting is indeed identical to the process of finding a global (or asymptotically global) minimum of $h$, and therefore we can also say that regularized models tend to actually effectively avoid configurations $\theta$ that lead to absolute minima of $h$ for a given input set $a$ and objective function $J$.

### Gradients are sensitive to minibatch composition

This idea runs somewhat against the current grain of intuition by many researchers, so reading the preceding paragraphs was probably not sufficient to convince an active member in the deep learning field.  But happily the concept may be shown experimentally and with clarity.  Take a relatively small network being trained to approximate some defined (and non-stochastic) function.  This practically any non-trivial function, and here we will focus on the example detailed [on this page](https://blbadger.github.io/neural-networks3.html). In this particular case, a network is trained to regress an output to approximate the function

$$
y = 10d
$$

where $y$ is the output and $d$ is one of 9 inputs.  The task is simple: the model must learn that $d$ is what determines the output, and must also learn to decipher the numerical input of $d$, or in other words the network needs to learn how to read numbers that are given in character form.  A modest network of 3.5 million parameters across 3 hidden layers is capable of performing this task extremely accurately. 

In the last section, the landscape of $h$ was considered.  Here we will focus on the gradient of $h$, as stochastic gradient descent is not affected by the values of $h$ but only their rate of change as $\theta$ chages. We can observe the gradient of the objective function $J(O(a; \theta))$ with respect to certain trainable parameters, say two parameters in vector form $x = (x_1, x_2)$.  The gradient is signified by $\nabla_x J(O(a; \theta))$ and resulting vector is two dimensional and may be plotted in the plane, as $x$ is equivalent to the projection of the gradient $\nabla_\theta J(O(a; \theta))$ onto our two parameters.  But we are interested in more than just the gradient of the parameters: we also want to visualize the landscape of the possible gradients nearby, that is, the gradients of $\nabla_x J(O(a; \theta))$ if we were to change the parameter $x$ slightly, as this is how learning takes place during SGD.  The gradient landscape may be plotted by assigning gradients to points on a 2-dimensional grid of possible values for the parameters $(x_1 + \epsilon_n, x_2 + \epsilon_n)\; n \in \Bbb Z$ that are near the model's true parameters $x$.  In the following plot, the $\nabla_x J(O(a; \theta))$ vector is
located at the center circle, and the surrounding vectors are the gradients $\nabla_x J(O(a; \theta))$ with $x$ modified to be $x_1+\epsilon_n, x_2 + \epsilon_n$ 

![gradients]({{https://blbadger.github.io}}/neural_networks/gradient_quiver.png)

Because our model is learning to approximate a deterministic function applied to each input, the classical view of stochastic gradient descent suggests that different subsets of our input set will give approximately the same gradient vectors for any given parameters, as the information content in each example is identical (the same rule is being applied to generate an output). In contrast, our idea is that we should see significant differences in the gradient vectors depending on the exact composition of our inputs, regardless of whether or not their informational content is identical w.r.t. the loss function.

Choosing an epoch that exhibits a decrease in the cost function $J(O(a; \theta))$ (corresponding to 6 seconds into [this video](https://www.youtube.com/watch?v=KgCuK6v_MgI)) allows us to investigate the sensitivity (or lack thereof) of the model's gradients to input $a$ during the learning process. As above the gradient's projection onto $(x_1, x_2)$ is plotted but now we observe the first two bias parameters in two hidden layers.  The model used on this page has three hidden layers, indexed from 0, and we will observe the gradient vectors on the second and third layer.

One can readily see that for 50 different minibatches $a_1, a_2,...,a_{50} \in a$ (each of size 64) of the same training set, there are quite different (sometimes opposite) vectors of $\nabla_x J(O(a_n; \theta))$ 

![gradients]({{https://blbadger.github.io}}/neural_networks/gradients_epoch10_eval.gif)

In contrast, at the start of training the vectors of $\nabla_x J(O(a; \theta))$ tend to yield gradients on $x$ that are (somewhat weak) approximations of each other.

![gradients]({{https://blbadger.github.io}}/neural_networks/gradients_start_eval.gif)

Regularization is the process of reducing the test error without necessarily reducing training error, and is thus important for overfitting.  One nearly ubiquitous regularization strategy is dropout, which is where individual neurons are stochastically de-activated during training in order to force the model to learn a family of closely related functions rather than only one.  It might be assumed that dropout prevents this difference in $\nabla_x J(O(a; \theta))$ between minibatches during training, but we see the opposite: instead, dropout leads to extremely unstable gradient vectors

![gradients]({{https://blbadger.github.io}}/neural_networks/gradients_epoch10.gif)

but once again this behavior is not as apparent at the start of training

![gradients]({{https://blbadger.github.io}}/neural_networks/gradients_start.gif)

Another technique used for regularization is batch normalization.  This method is motivated by an intrinsic problem associated with deep learning: the process of finding the gradient of the cost function $J$ with respect to parameters $x$ with respect to the cost function $\nabla_x J(O(a; \theta))$ may be achieved using backpropegation, but the gradient descent update of $x$, specifically $x - \epsilon\nabla_x J(O(a; \theta))$, assumes that no other parameters have been changed.  In a one-layer (inputs are connected directly to outputs) network this is not much of a problem because the contribution of $x_n$ (ie the weights) to the output's activations are additive. This is due to how most deep learning models are set up: in a typical case of a fully connected layer $h$ following layer $h_{-1}$ given the weight vector for that neuron $w$ and bias scalar $b$

$$
h = w^Th_{-1} + b
$$

where $w^Th_{-1}$ is a vector dot product, a linear transformation that adds all $w_nh_{-1,n}$ elements.  The gradient is computed and updated in (linear) vector space, so if a small enough $\epsilon$ is used then gradient descent should decrease $J$, assuming that computational round-off is not an issue.

But with more layers, the changes to network components becomes exponential with respect to the activations at $h$. To see why this is, note that for a four-layer network with biases set to 0 and weight vectors all equal to $w$

$$
h = w^T(w^T(w^T(w^Th_{-4})))
$$

Now updates to these weight vectors, $w - \epsilon\nabla_w J(O(a; \theta))$ are no longer linear with respect to the activation $h$.  In other words, depending on the values of the components of the model a small increase in one layer may lead to a large change in other layers' activations, which goes against the assumption of linearity implicit in the gradient calculation and update procedure.

Batch normalization attemps to deal with this problem by re-parametrizing each layer to have activations $h'$ such that they have a defined standard deviation of 1 and a mean of 0, which is accomplished by using the layer's activation mean $\mu$ and standard deviation $\sigma$ values that are calculated per minibatch during training.  The idea is that if the weights of each layer form distributions of unit variance around a mean of 0, the effect of exponential growth in activations (and also gradients) is minimized.

But curiously, batch normalization also stipulates that back-propegation proceed through these values $\sigma, \mu$ such that they are effectively changed during training in addition to changing the model parameter. Precisely, this is done by learning new parameters $\gamma, \beta$ that transform a layer's re-paremetrized activations $h'$ defined by the function

$$
h'' = \gamma h' + \beta
$$

which means that the mean is multiplied by $\gamma$ before being added by $\beta$, and the standard deviation is multiplied by $\gamma$. This procedure is necessary to increase the ability of batch normalized models to approximate a wide enough array of functions, but it in some sense defeats the intended purpose of ameliorating the exponential effect, as the transformed layer $h''$ has a mean and standard deviation can drift from the origin and unit value substantially. Why then is batch normalization an effective regularizer?

Let's investigate by applying batch normalization to our model and observing the effect on the gradint landscape during training. When 1-dimensional batch normalization is applied to each hidden layer of our model above, we find at 10 epochs that $\


_x J(O(\theta; a))$ exhibits relatively unstable gradient vectors in the middle layer.  As we saw for dropout and non-regularized gradients, different minibatches have very different gradient landscapes.

![gradients]({{https://blbadger.github.io}}/neural_networks/gradients_epoch10_batchnorm.gif)

Thus we come to the interesting observation that batch normalization leads to a similar loss of stability in the gradient landscape that is seen for dropout. which in this author's opinion is a probable reason for its success as a regularizer (given dropout's demonstrated success in this area).  This helps explain why it was found that batch normalization and dropout are often able to substitute for each other in large models: it turns out that they have similar effects on the gradient landscape of hidden layers, although batch normalization in this case seems to be a more moderate inducement of this loss of stability.

Note that for each of the above plots, the model's parameters $\theta$ did not change between evaluation of different minibatches $a_n$, of in symbols there is an invariant between $\nabla_x J(O(a_n; \theta)) \forall n$.  This means that the direction of stochastic gradient descent does indeed depend on the exact composition of the minibatch $a_n$.

To summarize, we find that the gradient with respect to four parameters can change drastically depending on the training examples that make of the given minibatch $a_n$.  As the network parameters are updated between minibatches, both the identity of the inputs per minibatch and the order in which the same inputs are used to update a network determine the path of stochastic gradient descent. This is why the identity of the input $a$ is so important, even for a fixed dataset with no randomness.

### Fashion MNIST

Fluorescent images of cells are unlikely to be met with in everyday life, unless you happen to be a biologist.  What about image classification for these objects, can the neural net architectures presented here learn these too?

The [fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) is a set of 28x28 monocolor images of articles of 10 types of clothing, labelled accordingly.  Because these images are much smaller than the 256x256 pixel biological images above, the architectures used above must be modified (or else the input images must be reformatted to 256x256).  The reason for this is because max pooling (or convolutions with no padding) lead to reductions in subsequent layer size, eventually resulting in a 0-dimensional layer.  Thus the last four max pooling layers were removed from the deep network, and the last two from the AlexNet clone ([code](https://github.com/blbadger/neural-network/blob/master/fmnist_bench.py) for these networks).  

The deep network with no other modifications than noted above performs very well on the task of classifying the fashion MNIST dataset, and >91 % accuracy rate on test datasets achieved with no hyperparameter tuning. 

![fashion MNIST]({{https://blbadger.github.io}}/neural_networks/Fashion_mnist.png)

AlexNet achieves a ~72% accuracy rate on this dataset with no tuning or other modifications, although it trains much slower than the deep network as it has many more parameters (over ten million in this case) than the deep network (~180,000).

We may observe a model's attribution on the inputs from this dataset as well in order to understand how a trained model arrives at its conclusion. Here we have our standard model architecture and we compute the gradientxinput

$$
\nabla_a O(a; \theta) * a
$$

where the input $a$ is a tensor of the input image (28x28), the output of the model with parameters $\theta$ and input $a$ is $O(a; \theta)$, and $*$ denotes Hadamard (element-wise) multiplication.  This may be accomplished using Pytorch in a similar manner as above, and for some variety we can also perform the calculation using Tensorflow as follows:

```python
def gradientxinput(features, label):
	...
	optimizer = tf.keras.optimizers.Adam()
	features = features.reshape(1, 28, 28, 1)
	ogfeatures = features # original tensor
	features = tf.Variable(features, dtype=tf.float32)
	with tf.GradientTape() as tape:
		predictions = model(features)

	input_gradients = tape.gradient(predictions, features).numpy()
	input_gradients = input_gradients.reshape(28, 28)
	ogfeatures = ogfeatures.reshape(28, 28)
	gradxinput = tf.abs(input_gradients) * ogfeatures
	...
```
such that `ogfeatures` and `gradxinput` may be fed directly into `matplotlib.pyplot.imshow()` for viewing.  Note that the images and videos presented here were generated using pytorch, with a similar (and somewhat less involved) implementation as the one put forth in the preceding section for obtaining gradientxinput tensors. 

For an image of a sandal, we observe the follopwing attribution:

![fashion MNIST gradientxinput]({{https://blbadger.github.io}}/neural_networks/fmnist_gradxinput.png)

which focuses on certain points where the sandal top meets the sole.  How does a deep learning model such as our convolutional network learn which regions of the input to focus on in order to minimize the cost function?  At the start of training, there is a mostly random gradientxinput attribution for each image

![fashion MNIST gradientxinput]({{https://blbadger.github.io}}/neural_networks/fmnist_attribution_grid0001.png)

but at the end of training, certain stereotypical features of a given image category receive a larger attribution than others: for example, the elbows and collars of coats tend to exhibit a higher attribution than the rest of the garment.

![fashion MNIST gradientxinput]({{https://blbadger.github.io}}/neural_networks/fmnist_attribution_grid0505.png)

It is especially illuminating to observe how attribution changes after each minibatch gradient update.  Here we go from the start of the start to the end of the training as show in the preceding images, plotting attributions on a subset of test set images after each minibatch (size 16) update.

{% include youtube.html id='7SCd5YVYejc' %}

### Flower Type Identification

For some more colorful image classifications, lets turn to Alexander's flower [Kaggle photoset](https://www.kaggle.com/alxmamaev/flowers-recognition), containing images of sunflowers, tulips, dandelions, dasies, and roses.  The deep network reaches a 61 % test classification score, which increases to 91 % for binary discrimination between some flower types. 

Examples of the deep network classifying images of roses or dandelions,

![flower classes]({{https://blbadger.github.io}}/neural_networks/Figure_flowers2.png)

sunflowers or tulips,

![flower classes]({{https://blbadger.github.io}}/neural_networks/Figure_flowers1.png)

and tulips or roses

![flower classes]({{https://blbadger.github.io}}/neural_networks/Figure_flowers_tulips_roses2.png)

We can investigate the learning process by using gradientxinput attribution. Before the start of training, we see that there is relatively random attribution placed on various pixels of test set flower images

![flower saliency]({{https://blbadger.github.io}}/neural_networks/flower_attributions0000.png)

but after 25 epochs, certain features are focused upon

![flower saliency]({{https://blbadger.github.io}}/neural_networks/flower_attributions1200.png)

Plotting attribution after every minibatch update to the gradient, we have

{% include youtube.html id='lVcNSD0viX0' %}

The deep learning models generally perform worse on flower type discrimination when they are not given color images, which makes sense being that flowers are usually quite colorful.  Before the start of training, we have a stochastic attribution: note how the model places relatively high attribution on the sky in the bottom three images (especially the bottom right)

![flower saliency]({{https://blbadger.github.io}}/neural_networks/colorflower_attributions0000.png)

In contrast, after 25 epochs of training the model has learned to place more attribution on the tulip flower body, the edge of the rose petals, and the seeds of the sunflower and dandelion.  Note that the bottom center tulip has questionable attribution: the edge of the leaves may be used to discriminate between plant types, but it is not likely that flower pot itself is a useful feature to focus on.

![flower saliency]({{https://blbadger.github.io}}/neural_networks/colorflower_attributions1202.png)

Plotting attribution after every minibatch update to the gradient, we have

{% include youtube.html id='mz_Qo1fcmgc' %}

### Adversarial Examples

Considering the attribution patterns placed on various input images, it may seem that a deep learning object recognition process is similar to a human-like decision making process when identifying images: focus on the features that differ between images and learn which features correspond to what image. But there are significant differences between natural and deep learning-based object recognition, and one of the most dramatic of these differences is the presence of what has been termed 'adversarial examples', first observed by Szegedy and colleagues in their [paper](https://arxiv.org/abs/1312.6199) on this subject.

To those of you who have read [this page](https://blbadger.github.io/nn-limitations.html) on the subject, the presence of adversarial examples should come as no surprise: as a model becomes able to discriminate between more and more input images it better and better approximates a one-to-one mapping between a multidimensional input (the image) and a one-dimensional output (the cost function).  

How might we go about finding an adversarial example?  One option is to compute the gradient $g$ of the loss function of the output $J(O)$ with respect to the input $a$ of the model with parameters $\theta$,

$$
g = \nabla_a J(O((a; \theta)))
$$

but instead of taking a small step against the gradient (as would be the case if we were computing $\nabla_{\theta} J(O(a; \theta))$ and then taking a small step in the opposite direction during stochastic gradient descent), we first find the direction along each input tensor element that $g$ projects onto with $\mathrm{sign}(g)$ and then take a small step in this direction.

$$
a' = a + \epsilon * \mathrm{sign}(g)
$$

where the $\mathrm{sign}()$ function the real-valued elements of a tensor $a$ to either 1 or -1, or more precisely this function is $\Bbb R \to {-1, 1}$ depending on the sign of each element $a_n \in a$. 

What this procedure accomplishes is to change the input by a small amount (determined by the size of $\epsilon$) in the direction that makes the cost function $J$ increase the most, which intuitively is effectively the same as making the input slightly different in precisely the way that makes the neural network less accurate.  

To implement this, first we need to calculate $g$, which may be accomplished as follows:

```python
def loss_gradient(model, input_tensor, true_output, output_dim):
	... # see source code for the full method with documentation
	true_output = true_output.reshape(1)
	input_tensor.requires_grad = True
	output = model.forward(input_tensor)
	loss = loss_fn(output, true_output) # objective function applied (negative log likelihood)

	# only scalars may be assigned a gradient
	output = output.reshape(1, output_dim).max()

	# backpropegate output gradient to input
	loss.backward(retain_graph=True)
	gradient = input_tensor.grad
	return gradient
```

Now we need to calculate $a'$, and here we assign $\epsilon=0.01$. Next we find the model's output for $a$ as well as $a'$, and throw in an output for $g$ for good measure

```python
def generate_adversaries(model, input_tensors, output_tensors, index):
	... # see source code for the full method with documentation
	single_input= input_tensors[index].reshape(1, 3, 256, 256)
	input_grad = torch.sign(loss_gradient(model, single_input, output_tensors[index], 5))
	added_input = single_input + 0.01*input_grad
	
	original_pred = model(single_input)
	grad_pred = model(0.01*input_grad)
	adversarial_pred = model(added_input)
```

Now we can plot images of $a$ and the output of each fed to the model by reshaping the tensor for use in `plt.imshow()` before finding the output class and output confidence (as we are using a softmax output) from $O(a; \theta)$ as follows

```python
	...
	input_img = single_input.reshape(3, 256, 256).permute(1, 2, 0).detach().numpy() # reshape for imshow 
	original_class = class_names[int(original_pred.argmax(1))].title() # find output class
	original_confidence = int(max(original_pred.detach().numpy()[0]) * 100) # find output confidence w/softmax output
```

and finally we can perform the same procedure to yield $g$ and $a'$.  

For an untrained model with randomized $\theta$, $O(a';\theta)$ is generally quite similar to $O(a;\theta)$. The figures below display typical results from computing $\mathrm{sign}(g)$ (center) from the original input $a$ (left) with the modified input $a'$ (right). Note that the image representation of the sign of the gradient clips negative values (black pixels in the image) and is not a true representation of what is actually added to $a$: the true image of $\mathrm{sign}(g)$ is 50 times dimmer than shown (meaning $\epsilon = 0.01$), and by design nothing is visible. The model's output for $a$ and $a'$ is noted above the image, with the softmax value converted to a percentage for clarity.

![adversarial example]({{https://blbadger.github.io}}/neural_networks/flower_start_adversarial0.png)

![adversarial example]({{https://blbadger.github.io}}/neural_networks/flower_start_adversarial1.png)

After training, however, we see some dramatic changes in the model's output (and ability to classify) the image $a$ compared to the shifted image $a'$. In the following three examples, the model's classification for $a$ is accurate, but the addition of an imperceptably small amount of the middle image to make $a'$ yields a confident but incorrect classification.

![adversarial example]({{https://blbadger.github.io}}/neural_networks/adversarial_example0.png)

![adversarial example]({{https://blbadger.github.io}}/neural_networks/adversarial_example1.png)

![adversarial example]({{https://blbadger.github.io}}/neural_networks/adversarial_example6.png)

Not all shifted images experience this change in predicted classification: the following images are viewed virtually identically by the model.  After 40 epochs of training a cNN, around a third of all inputs follow this pattern such that the model does not change its output significantly when given $a'$ as an input instead of $a$.

![adversarial example]({{https://blbadger.github.io}}/neural_networks/adversarial_example13.png)

It is interesting to note that the gradient sign image itself may be confidently (and necessarily incorrectly) classified too.

![adversarial example]({{https://blbadger.github.io}}/neural_networks/gradpred_adversarial_example3.png)

Can we find adversarial examples for simpler inputs as well as complicated ones? Indeed we can: after applying the gradient step method to 28x28 pixel Fashion MNIST images using a model trained to classify these inputs, we can find adversarial examples just as we saw for flowers.

![adversarial example]({{https://blbadger.github.io}}/neural_networks/fmnist_adversarial_example.png)


