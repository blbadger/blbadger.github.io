## Image classification with [neural networks](https://github.com/blbadger/neural-network).

### Introduction

In scientific research it is often important to be able to categorize samples accurately. For example, during the research process a histological section of tissue from an animal subjected to some treatment may be observed to be healthy or unhealthy depending on its appearance under the microscope. One way to accomplish this task of classification of healthy versus unhealthy is to have a skilled individual manually observe and interpret these images.  With experience, the human eye is in many instances able to accurately discern and categorize many important features from microsopic observation.

Another way to accomplish the task is to employ a computational method.  One could imagine attempting to program a machine to recognize the same features that an expert individual would take note of in order to replicate that individual's performance.  Historically this approach was taken in efforts resulting in what were called 'expert systems', but now such approaches are generally no longer attempted.  

This is for a couple reasons: firstly because they are extremely tedious and time-consuming to implement, and secondly because once implemented these methods generally fail to be very accurate. The underlying cause for both of these points of failure is that it is actually very difficult to explain complicated tasks like image recognition precisely yet generally enough to be accurate.  

The failures of attempts to directly replicate complex tasks programatically has led to increased interest in the use of probability theory to make estimations rather than absolute statements, resulting in the expansion of a field that has been termed statistical learning.  Particularly delicate tasks like image classification often require particularly detailed computational methods for best accuracy, and this computationally-focused statistical learning is called machine learning.

Supervised machine learning techniques attempt to reproduce with generalization: in the case of image classification, we already 'know' how to classify a certain number of images that becomes our training dataset, and we seek to classify other images that the program has not seen before.  In order to assess accuracy for classification on this test dataset, we need to know the true classification to be able to compare this with the predicted classification.  The ideal supervised learning technique takes a training dataset and accurately generalizes from this such that unseen datasets are classified correctly using information from this training dataset.  


### Designing a neural network to classify images of cells

The current state-of-the-art method for classifying images is achieved with neural networks.  These are programs that use nodes called artificial 'neurons' that have associated weights and biases that are tuned in accordance to some loss function in order to 'learn'.  These neuron are arranged in sequential layers, each representing the input in a potentially more abstract manner before the output layer is used for classification.  This sequential (and distributed across many neurons of one layer) representation may be many layers deep before reaching the output, leading to the field of study for many neural network-related approaches to be called 'deep learning'.

A good hands-on introduction to the theory and utility of neural networks for image classification is found in Nielsen's [book](http://neuralnetworksanddeeplearning.com/), and the core algorithms of stochastic gradient descent and backpropegation that are used to train neural nets on this page are explained there.  For a deeper and more comprehensive study of this and related topics, perhaps the best resource is the [classic text](https://www.deeplearningbook.org/) by Goodfellow, Bengio, and Courville.  This page will continue with the assumption of some familiarity with the fundamental ideas behind deep learning.

Neural networks are by no means simple constructions, but are not so prohibitively complicated that they cannot be constructed from scratch such that each operation to every individual neuron is clearly defined (see [this repo](https://github.com/blbadger/neural-network/blob/master/connected/fcnetwork_scratch.py) for an example).  But this approach is relatively slow: computing a matrix (usually called an n-dimensional array or a tensor) multiplication or else Hadamard product element-wise takes far longer than broadcasted computation with arrays.  In python, numpy is perhaps the most well-known and powerful library for general matrix manipulation is numpy, and this library can be used to simplify and speed up the network linked earlier in this paragraph: [here](https://github.com/blbadger/neural-network/blob/master/connected/fcnetwork.py) is an implementation.  In essence, training and evaluating feed-forward neural networks by manipulating matrix elements changes the weights and biases of many 'neurons' simultaneously, which is far faster than tuning each individual neurons as above.  

As effective as numpy is, it is not quite ideal for speedy computations with large arrays, specifically because it does not optimize memory allocation and cannot make use of a GPU for many simultaneous calculations. To remedy this situation, there are a number of libraries were written (mostly) for the purpose of rapid computations with the tensors, with Tensorflow and PyTorch being the most popular.  I decided to try Tensorflow (with a Keras front end) and found a number of docs extremely useful in setting up a neural network, perhaps the most accessible being the official [Keras introduction](https://www.tensorflow.org/tutorials/keras/classification) and [load image](https://www.tensorflow.org/tutorials/load_data/images?hl=TR) tutorials.  

Let's set up the network!  The first thing to do is to prepare the training and test datasets.  There are datsets online with many diverse images that have been prepped for use for neural network classification, but if one wishes to classify objects in images one takes, then preparation is necessary.  The primary thing that is necessary is to label the images such that the neural network knows which image represents which category.  With Keras this can be accomplished by simply putting all images of one class in one folder and putting images of another class in another folder, and then by saving the directory storing both folders as a variable before calling the appropriate data loading functions.

Say you want to train a network to classify images of objects seen from a moving car.  One way of doing this is by splitting up large wide-field images that contain the whole street view into smaller pieces that have at most one object of interest.  Neural networks work best with large training sets composed of thousands of images, so I took this approach to split up images of fields of cells into many smaller images of one or a few cells.  We can also performed a series of rotations on these images, resulting in many copies of each original image at different rotations.  This is a common tactic for increasing training efficacy for neural networks, and can also be performed directly using Tensorflow (preprocessing module).

```python
from PIL import Image
import image_slicer
import os

image_files = ['path/to/file.jpg', 'path/to/file2.png']
image_directory = '/home/bbadger/Desktop/GFP_2/snap29'


class Imageprep:

	def __init__(self, image_file):
		self.image_files = image_files

	def slice(self, slices):
		"""
		Divides an image into smaller images of equal dimension (l//n by w//n)

		Args:
			slices: int, the number of desired smaller images

		Returns:
			None (appends sliced files to directory)

		"""

		for i, file in enumerate(self.image_files):
			im = Image.open(file)

			# save images as 'path/to/file_0_1.jpg' etc.
			image_slicer.slice('/path/to/file.jpg', slices)

		return
		

	def rename(self, image_directory):
		"""
		Rename files with a name that is easily indexed for future iteration.

		Args:
			image_directory: str, path to directory containing images

		Returns:
			None (modifies image names in-place)

		"""

		path = image_directory
		files = os.listdir(path)

		for index, file in enumerate(files):
			os.rename(os.path.join(path, file), os.path.join(path, ''.join(['s', str(index), '.png'])))

		return


	def rotate(self, image_directory):
		"""
		Rotate images from -5 to 5 degrees, 1 degree increments, saving each.

		Args:
			image_directory: str, path to directory of images

		Returns:
			None (saves images)

		"""

		path = image_directory
		files = os.listdir(path)

		for i, file in enumerate(files):
			im = Image.open(file)
			for j in range(-5,5):
				im.rotate(j)

				# reformat file name (assumes . only in file ending)
				pair_ls = [i for i in file.split('.')]
				core_name = pair_ls[0]
				ending = pair_ls[1]
				new_file_name = core_name + f'_{i}_{j}' + ending

				im.save(new_file_name)

		return

```


Now let's design the neural network.  We shall be using established libraries for this set, so reading the documentation for [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/api_docs) may be useful.  First we write a docstring for our program stating its purpose and output, and import the relevant libraries.


```python

"""
Tensorflow_sequential_deep.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Implementation of a Keras sequential neural network using a 
Tensorflow backend, combined with modules to display pre- classified
images before the network trains as well as a subset of test images
with classification and % confidence for each image. 
"""

### Libraries
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
prints out the number of files and the names of the folders in the `data_dir` directory of choice.  If the number of files does not match what is observed by listing in terminal, or if there are any folders or files with the `._.DS_Store` ending then one strategy is to simply copy all files of the relevant directory into a new folder and check again.

Now the images can be rescaled (if they haven't been already), as all images will need to be the same size. Here all images are scaled to a heightxwidth of 256x256 pixels.  Then a batch size is specified.  This number determines the number of images seen for each training epoch.

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

The same process is repeated for the other directories, which in this case contain test image datasets.

```python
	CLASS_NAMES = np.array([item.name for item in data_dir2.glob('*') 
				if item.name not in ['._.DS_Store', '.DS_Store', '._DS_Store']])

	test_data_gen1 = image_generator.flow_from_directory(
	    directory=str(data_dir2), 
	    batch_size=783, 
	    shuffle=True, 
	    target_size=(IMG_HEIGHT,IMG_WIDTH),
	    classes=list(CLASS_NAMES))


	CLASS_NAMES = np.array([item.name for item in data_dir3.glob('*') 
				if item.name not in ['._.DS_Store', '.DS_Store', '._DS_Store']])

	test_data_gen2 = image_generator.flow_from_directory(
	    directory=str(data_dir3), 
	    batch_size=719, 
	    shuffle=True, 
	    target_size=(IMG_HEIGHT,IMG_WIDTH),
	    classes=list(CLASS_NAMES))
	    
	return train_data_gen1, test_data_gen1, test_data_gen2
```

An image set generation may be checked with a simple function that plots a subset of images. This is particularly useful when expanding the images using translations or rotations or other methods in the `image_generator` class, as one can view the images after modification.

```python
def show_batch(image_batch, label_batch):
    """
    Takes a set of images (image_batch) and an array of labels (label_batch)
    from the tensorflow preprocessing image_generator as arguments and subsequently
    returns a plot of a 25- member subset of the image set specified, complete
    with a classification label for each image.  
    	
    """
    
    plt.figure(figsize=(10,10))
    
    for n in range(25):
        ax = plt.subplot(5, 5, n+1)
        plt.imshow(image_batch[n])
        plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
        plt.axis('off')
	
    plt.show()
    plt.close()
    return


### calls show_batch on a preprocessed dataset to view classified images
train_data_gen1, test_data_gen1, test_data_gen2 = data_import()
image_batch, label_batch = next(train_data_gen1)
show_batch(image_batch, label_batch)
```

Assigning the pair of labels to each iterable in the relevant generators,

```python
(x_train, y_train) = next(train_data_gen1)
(x_test1, y_test1) = next(test_data_gen1)
(x_test2, y_test2) = next(test_data_gen2)
```

Now comes the fun part: assigning a network architecture! The Keras `Sequential` model is a straightforward, if relatively limited, class that allows a sequential series of network architectures to be added into one model. This does not allow for branching or recurrent architectures, in which case the more flexible functional Keras `tensorflow.keras.Model` should be used.  Below is an example of the network implemented using `keras.Model`.  After some trial and error, the following architecture was found to be effective for the relatively noisy images here.

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

Now the network can be compiled, trained, and evaluated on the test datasets. I also include `model.summary()` in order to have the specific model printed, if one is interested to learn exactly how many parameters each layer contains.

```python
model.compile(optimizer='Adam', 
	loss = 'categorical_crossentropy', 
	metrics=['accuracy'])

model.summary()
model.fit(x_train, y_train, epochs=9, batch_size = 20, verbose=2)
model.evaluate(x_test1, y_test1, verbose=1)
model.evaluate(x_test2, y_test2, verbose=1)

```

A few notes about this architecture: first, the output is a softmax layer and therefore yields a probability distribution for an easy-to-interpret result.  The data labels are one-hot encoded, meaning that the label is denoted by a vector with one 'hot' label (usually 1), ie instead of labels such as `[3]` we have `[0, 0, 1]`.  This means that categorical crossentropy should be used instead of sparse categorical crossentropy.  

Another thing to note is the lack of normalization: there are no batch normalizations applied to any layers, no dropout nor even L1 or L2 normalization applied to neuron weights.  As we shall see below, this does not prevent the network from achieving very high test classification accuracy, which seems to go against the conventional wisdom for neural network architecture.  As an explanation for this, first note that convolutional layers have intrinsic protection against overfitting, as they are translation-insensitive.  Second, adaptive moment estimation (Adam) is employed as our optimization method, and this algorithm was specifically designed to converge for stochastic and sparse objective function outputs.  It has become clear that normalizations such as batch norm do not act to reduce internal covariant shift, but instead acts to smooth out the objective function.  With Adam, the objective function does not need to be smooth for effective tuning of weights and biases.  Finally, dataset augmentation via rotations and translations provides another form of insurance against overfitting.  Finally and most importantly, training is abbreviated to avoid overfitting.

### The network accurately classifies test set images

Using blinded approach, I classified 93 % of images correctly for certain test dataset, which we can call 'Snap29' after the gene name of the protein that is depleted in the cells of half the dataset (termed 'Snap29') along with cells that do not have the protein depleted ('Control').  There is a fairly consistent pattern in these images that differentiates 'Control' from 'Snap29' images: depletion leads to the formation of aggregates of fluorescent protein in 'Snap29' cells.

The network shown above averaged ~96 % accuracy (over a dozen training runs) for this dataset.  We can see these test images along with their predicted classification ('Control' or 'Snap29'), the confidence the trained network ascribes to each prediction, and the correct or incorrect predictions labelled green or red, respectively.  The confidence of assignment is the same as the activation of the neuron in the final layer representing each possibility.  This can be achieved as follows:

```python
### Creates a panel of images classified by the trained neural network.

image_batch, label_batch = next(test_data_gen1)

test_images, test_labels = image_batch, label_batch

predictions = model.predict(test_images)

def plot_image(i, predictions, true_label, img):
    """ 
    Returns a test image with a predicted class, prediction
    confidence, and true class labels that are color coded for accuracy.
    """
    prediction, true_label, img = predictions[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    predicted_label = np.argmax(predictions)
    if prediction[0] >=0.5 and true_label[0]==1:
        color = 'green'
    elif prediction[0] <=0.5 and true_label[0]==0:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("{} % {}, {}".format(int(100*np.max(prediction)), 
    'Snap29' if prediction[0]>=0.5 else 'Control', 
    'Snap29' if true_label[0]==1. else 'Control'), color = color)

    return

num_rows = 4
num_cols = 3
num_images = 24

# Plot initialization
plt.figure(figsize = (num_rows, num_cols))

# Plot assembly and display
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, i+1)
    plot_image(i+1, predictions, test_labels, test_images)

plt.show() 

```

which yields

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/nn_images_1.png)

Let's see what happens when the network is applied to an image set without a clear difference between the 'Control' and experimental group (this time 'Snf7', named  after the protein depleted from these cells in this instance).  After being blinded to the true classification labels, I correctly classified 70.7 % of images of this dataset.  This is better than chance (50 % classification accuracy being binary) but much worse than for the Snap29 dataset. Can the neural network do better?

It cannot: the average training run results in 62 % classification accuracy, and the maximum accuracy achieved was 66 %, both slightly lower than my manual classification accuracy.  We can see the results of one particular training run: the network confidently predicts the classification of nearly all images, but despite this confidence is incorrect on the identity of many.

![snf7 test accuracy]({{https://blbadger.github.io}}/neural_networks/nn_images_2.png)

Each time a network is trained, there is variability in how effective the training is even with the same datasets as inputs.  Because of this, it is helpful to observe a network's performance over many training runs (each run starting with a naive network and ending with a trained one)  The statistical language R (with ggplot) can be used to make a box plot of the test accuracies achieved over many training runs for these datasets, once this data has been saved as text file. Here I use a csv file in the directory shown to compare the test accuracies of Snap29 compared to Snf7 datasets

```R
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

which yields

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/nn_images_3.png)


### Overfitting (memorization) versus learning

Why does the network confidently predict incorrect answers for the Snf7 dataset?  Let's see what happens during training.  One way to gain insight into neural network training is to compare the accuracy of training image classification at the end of each epoch.  This can be done in R as follows:

```R
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

As backpropegation lowers the cost function, we would expect for the classification accuracy to increase for each epoch trained. For the network trained on the Snap29 dataset, this is indeed the case: the average training accuracy increases in each epoch and reaches ~94 % after the last epoch.  But something very different is observed for the network trained on Snf7 images: a rapid increase to 100 % training accuracy by the third epoch is observed, and subsequent epochs maintain the 100 % accuracy.

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/nn_images_4.png)

The high training accuracy (100%) but low test accuracy (62 %) for the network on Snf7 dataset images is indicative of a phenomenon in statistical learning called overfitting.  Overfitting signifies that a statistical learning procedure is able to accurately fit the training data but is not generalizable such that it fails to achieve such accuracy on test datasets.  Statistical learning models (such as neural networks) with many degrees of freedom are somewhat prone to overfitting because small variations in the training data are able to be captured by the model regardless of whether or not they are important for true classification.  

Overfitting is intuitively similar to memorization: both lead to an assimilation of information previously seen, without this assimilation necessarily helping the prediction power in the future.  One can memorize exactly how letters look in a serif font, but without the ability to generalize then one would be unable to indentify many letters in a sans-serif font.  The goal for statistical learning is to make predictions on hitherto unseen datasets, and thus to avoid memorization which does not guarantee this ability.

Thus the network overfits Snf7 images but is able to effectively differentiate Snap29 images.  This makes sense from the perspective of manual classification as there is a relatively clear pattern that one can use to distinguish Snap29 images, but little pattern to identify Snf7 images.  

A slower learning process for generalizable learning is observed compared to that which led to overfitting, but perhaps some other feature could be causing this delay in training accuracy.  The datasets used are noticeably different: Snap29 is in two colors whereas Snf7 is monocolor.  If a deeper network (with the extra layer of 50 dense neurons before the output layer) is trained on a monocolor version of the Snap29 or Snf7 datasets, the test accuracy achieved is nearly identical to what was found before,

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/nn_images_5.png)

And we see that this deeper network continues to confidently predict incorrect classifications for Snf7

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/nn_images_6.png)

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/nn_images_7.png)

Once again Snap29 training accuracy lags behind that of Snf7.

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/nn_images_8.png)

### AlexNet revisited

To see if the faster increase in training accuracy for Snf7 was peculiar to the particular network architecture used, I designed a network that mimics the groundbreaking architecture now known as [AlexNet](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), and the code to do this may be found [here](https://github.com/blbadger/neural-network/blob/master/AlexNet_sequential.py).  There are a couple differences between my recreation and the original that are worth mentioning: first, the original was split across two machines for training, leading to some parallelization that was not necessary for my training set.  More substantially, AlexNet used a somewhat idiosyncratic normalization method that is related to but distinct from batch normalization, which has been substituted here.  Finally, the output layer has only two rather than many neurons, as there are two categories of interest here.

Using this network, it has previously been seen that overfitting is the result of slower increases in training accuracy relative to general learning (see [this paper](https://arxiv.org/abs/1611.03530) and [another paper](https://dl.acm.org/doi/10.5555/3305381.3305406)).  With the AlexNet clone, once again the training accuracies for Snf7 increased faster than for Snap29 (although test accuracy was poorer for both relative to the deep network above).  This suggests that faster training leading to overfitting in the Snf7 dataset is not peculiar to one particular network architecture and hyperparameter choice.

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/nn_images_10.png)

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/nn_images_11.png)

There are a number of interesting observations from this experiment.  Firstly, the multiple normalization methods employed by AlexNet (relative to no normalization used for our custom architecture) are incapable of preventing severe overfitting for Snf7, or even for Snap29.  Second, with no modification to hyperparameters the AlexNet architecture was able to make significant progress towards classifying Snap29 images, even though this image content is far different from the CIFAR datasets that AlexNet was designed to classify.  The third observation is detailed in the next section.

### Learning (ie increasing test classification accuracy) does not equate to global minimization of a cost function during training

The above experimental results are quite interesting because they suggest that, for deep learning models of quite different architecture, a model's ability to generalize depends not only on the model's parameters $\theta$ and the family of functions that it can approximate given its architecture $F_j$ but also on the choice of input $x$.  Furthermore, the input $x$ is capable of inducing overfitting in a model employing extensive measures to prevent this phenomenon (L2 regularization, batch normalization) but a different input results in minimal overfitting (in the same epoch number) in a model that has none of these measures (nor any other regularizers separate from the model architecture itself).

Why is this the input so important to the behavior of the network?

Neural networks learn by adjusting the weights and biases of the neurons in order to miminize a cost function, which is a continuous and differentiable representation of the accuracy of the output of the network. This is usually done with a variant on stochastic gradient descent, which involves calculating the gradient (direction of largest change) using a randomly chosen subset of images (which is why it is 'stochastic').  This is conceptually similar to rolling a ball representing the network down a surface representing the multidimensional weights and biases of the neurons.  The height of the ball represents the cost function, so the goal is to roll it to the lowest spot during training.  

As we have seen in the case for Snf7 dataset images, this idea is accurate for increases in training accuracy but not not necessarily for general learning: reaching a global minimum for the cost function (100 % accuracy, corresponding to a cost function of less than 1E-4) led to severe overfitting and confident prediction of incorrect classifications.  This phenomenon of overfitting stemming from a global minimum in the cost function during training is not peculiar to this dataset either, as it has also been observed [elsewhere](http://proceedings.mlr.press/v38/choromanska15.pdf).  

If decreasing the neural network cost function is the goal of training, why would an ideal cost function decrease (to a global minimum) not be desirable?  In our analogy of a ball rolling down the hill, something important is left out: the landscape changes after every minibatch (more accurately, after every computation of gradient descent and change to neuronal weights and biases using backpropegation).  Thus as the ball rolls, the landscape changes, and this change depends on where the ball rolls. 

In more precise terms, for any given fixed neural network or similar deep learning approach we can fix the model architecture to include some set of parameters that we change using stochastic gradient descent to minimize some objective function.  The 'height' $h$ of our landscape is the value of the objective function,  $F$.

In this idealized scenario, the objective function $F$ is evaluated on an infinite number of input examples, but practically we can only evaluate it on a finite number of training examples.  $F$ evaluated on set $j$ of training examples is denoted $F_j$, and the argument for this function is the value and configuration of all model parameters, denoted $v_i$ such that the value of the objective function is

$$
h = F_j(v_i)
$$

What is important to recognize is that, in a non-idealized scenario, $h$ can take on many different values for any given model configuration $v_i$ depending on the specific training examples used to evaluate the objecting function $F_j$.  Thus the 'landscape' of $h$ changes depending on the order and identity of inputs $j$ fed to the model during training, even for a model of fixed architecture.  This is true for both the frequentist statistical approach in which there is some single optimal value of $v_i$ that minimizes $h = F_j(v_i)$ as well as the Bayesian statistical approach in which the optimal value of $v_i$ is a random variable as it is unkown whereas any estimate of $v_i$ using the data is fixed.

As $j$ is finite, an optimal value, ie a global minimum, of $h$ can be achieved by using stochastic gradient descent without any form of regularization, provided that the model has the capacity to 'memorize' the inputs $j$. This is called overfitting, and is strenuously avoided in practice because reaching this globabl minimum nearly always results in a model that performs poorly on data not seen during training.  But it is important to note that this process of overfitting is indeed identical to the process of finding a global (or asymptotically global) minimum of $h$, and therefore we can also say that one tends to actually strenuously avoid configurations $v_i$ that lead to absolute minima of $h$ for a given input set and objective function $F_j$


### Extensions to other datasets: fashion MNIST and flower types

Fluorescent images of cells are unlikely to be met with in everyday life, unless you happen to be a biologist.  What about image classification for these objects, can the neural net architectures presented here learn these too?

The [fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) is a set of 28x28 monocolor images of articles of 10 types of clothing, labelled accordingly.  Because these images are much smaller than the 256x256 pixel biological images above, the architectures used above must be modified (or else the input images must be reformatted to 256x256).  The reason for this is because max pooling (or convolutions with no padding) lead to reductions in subsequent layer size, eventually resulting in a 0-dimensional layer.  Thus the last four max pooling layers were removed from the deep network, and the last two from the AlexNet clone ([code](https://github.com/blbadger/neural-network/blob/master/fmnist_bench.py) for these networks).  

The deep network with no other modifications than noted above performs very well on the task of classifying the fashion MNIST dataset, and >91 % accuracy rate on test datasets achieved with no hyperparameter tuning. 

![fashion MNIST]({{https://blbadger.github.io}}/neural_networks/Fashion_mnist.png)

AlexNet achieves a ~72% accuracy rate on this dataset with no tuning or other modifications, although it trains much slower than the deep network as it has many more parameters (over ten million in this case) than the deep network (~180,000).

For some more colorful image classifications, lets turn to Alexander's flower [Kaggle photoset](https://www.kaggle.com/alxmamaev/flowers-recognition), containing images of sunflowers, tulips, dandelions, dasies, and roses.  The deep network reaches a 61 % test classification score, which increases to 91 % for binary discrimination between some flower types. 

Examples of the deep network classifying images of roses or dandelions,

![flower classes]({{https://blbadger.github.io}}/neural_networks/Figure_flowers2.png)

sunflowers or tulips,

![flower classes]({{https://blbadger.github.io}}/neural_networks/Figure_flowers1.png)

and tulips or roses

![flower classes]({{https://blbadger.github.io}}/neural_networks/Figure_flowers_tulips_roses2.png)
















