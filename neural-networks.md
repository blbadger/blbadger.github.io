## Image classification with a convolutional neural network

The full code for this section may be found [here](https://github.com/blbadger/neural-network).

Say that you were tasked with classifying images using an automated system, one that does not depend on user input.  There could be many reasons for this, and the one that motivated the making of this particular network was an effort to assess reproducibility.  I have taken many images of biological specimens (usually fluorescence images of cells) over the last decade, and think I have gotten pretty good at determining which images are of what category.  Can an automated system perform this task as good or better than me?

Programs that classify complicated things often fall into the category of machine learning.  Machine learning is a subset of supervised statistical learning techniques, some of which may be more familiar than others.  Statistical learning is the process of pattern identification in statistical data.  Linear regression, decision trees, and neural networks are all supervised statistical learning techniques, meaning that these techniques use information from a training dataset to predict the outputs of a test dataset.  

Supervised techniques are in the business of reproduction: in the case of image classification, we already 'know' how to classify a certain number of images that becomes our training dataset, and we seek to classify other images that the program has not seen before.  In order to assess accuracy for classification on this test dataset, we need to know the true classification to be able to compare this with the predicted classification.  The ideal supervised learning technique takes a training dataset and accurately generalizes from this such that unseen datasets are classified correctly using information from this training dataset. In essence, we are seeking to reproduce the previous classification of images in the training dataset.

The current state-of-the-art method for classifying images is with neural networks.  These are programs that use nodes called 'neurons' that change in accordance to the training dataset.  Perhaps the best introduction to neural networks is an online book by Neilsen found [here](http://neuralnetworksanddeeplearning.com/).  This book shows one how to set up a neural network from scratch in order to best understand what these programs to.  For speed and ease of modification, there are a number of libraries that exist for the purpose of making neural networks: Theano, Keras, Tensorflow, and PyTorch are a few for python.  I decided to try Tensorflow (which uses a Keras front end) and found a number of introductory materials extremely useful in setting up a neural network, perhaps the most accessible being the official [Keras introduction](https://www.tensorflow.org/tutorials/keras/classification) and [load image](https://www.tensorflow.org/tutorials/load_data/images?hl=TR) tutorials.  

Let's set up the network!  The first thing to do is to prepare the training and test datasets.  There are datsets online with many diverse images that have been prepped for use for neural network classification, but if one wishes to classify objects in images one takes, then preparation is necessary.  The primary thing that is necessary is to label the images such that the neural network knows which image represents which category.  Using Keras on a Tensorflow backend, this can be accomplished by simply putting all images of one class in one folder and putting images of another class in another folder, and then by saving the directory storing both folders as a variable.  

Say you want to train a network to classify images of objects seen from a moving car.  One way of doing this is by splitting up large wide-field images that contain the whole street view into smaller pieces that have at most one object of interest.  Neural networks work best with large training sets composed of hundreds if not thousands of images, so I took this approach to split up images of fields of cells into many smaller images of one or a few cells.  I then performed a series of rotations on these images, resulting in many copies of each original image at different rotations.  This is a common tactic for increasing training efficacy for neural networks, and can also be performed directly using Tensorflow (preprocessing module).  Here are the scripts I wrote to give you an idea of the particular process I used in python.

```python
### Image slicer- preps images for classification via NNs
from PIL import Image
import image_slicer

for i in range(1, 30):
	im = Image.open('/Users/badgerbl/Desktop/Control/Control_{}.jpg'.format(i))
	image_slicer.slice('/Users/badgerbl/Desktop/Control/Control_{}.jpg'.format(i), 16)
	

### renames files with a name that is easily indexed for future iteration
import os

path = '/home/bbadger/Desktop/GFP_2/snap29'
files = os.listdir(path)

for index, file in enumerate(files):
	os.rename(os.path.join(path, file), os.path.join(path, ''.join(['s', str(index), '.png'])))

	
### Image rotater- preps images for classification via NNs by rotation
from PIL import Image

for i in range(1, 348):
	im = Image.open('/Users/badgerbl/Desktop/unrotated/Control_{}.png'.format(i))
	for j in range(-5,5):
		im.rotate(j)
		im.save('/Users/badgerbl/Desktop/Control_rotated/Control_{}_{}.png'.format(i, j))
```


Next let's make the neural network program.  This will call on external APIs, so the documentation for [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/api_docs) is important to anyone wishing to set up a network on these libraries.  First we write a docstring for our program stating its purpose and output, and import the relevant libraries.


```python

"""
Tensorflow_sequential_deep.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Implementation of a Keras sequential neural network using a 
Tensorflow backend, combined with modules to display pre- classified
images before the network trains as well as a subset of test images
with classification and % confidence for each image.  The latter script
is optimized for binary classification but can be modified for more than
two classes.
"""

### Libraries
# Standard library
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import pathlib

# Third-party libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 

```

Now we can assign the directories storing the training and test datasets assembled above to variables that will be called later.  Here `data_dir` is our training dataset and `data_dir2` and `data_dir3` are test datasets, which link to folders I have on my Desktop.

```python
data_dir = pathlib.Path('/home/bbadger/Desktop/neural_network_images',  fname='Combined')
data_dir2 = pathlib.Path('/home/bbadger/Desktop/neural_network_images2', fname='Combined')
data_dir3 = pathlib.Path('/home/bbadger/Desktop/neural_network_images3', fname='Combined')
```

Now comes a troubleshooting step.  In both Ubuntu and MacOSX, I have encountered 

```python
image_count = len(list(data_dir.glob('*/*.png')))

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name not in ['._.DS_Store', '._DS_Store', '.DS_Store']])

print (CLASS_NAMES)
print (image_count)
```












