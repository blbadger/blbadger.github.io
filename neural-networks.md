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

Now comes a troubleshooting step.  In both Ubuntu and MacOSX, `pathlib.Path` sometimes recognizes folders or files ending in `._.DS_Store` or a variation on this pattern.  These folders are empty and appear to be an artefact of using `pathlib`, as they are not present if the directory is listed in a terminal, and these interfere with proper classification by the neural network.  To see if there are any of these phantom files,

```python
image_count = len(list(data_dir.glob('*/*.png')))

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name not in ['._.DS_Store', '._DS_Store', '.DS_Store']])

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

`image_generator` also has kwargs to specifiy rotations and translations to expand the training dataset, although neither of these functions are used in this example because the dataset has already been expanded via rotations.

```python
train_data_gen1 = image_generator.flow_from_directory(directory=str(data_dir),
	batch_size=BATCH_SIZE, shuffle=True, target_size=(IMG_HEIGHT,IMG_WIDTH), 
		classes=list(CLASS_NAMES), subset = 'training')
```

The same process is repeated for the other directories, which in this case contain test image datasets.

```python
CLASS_NAMES = np.array([item.name for item in data_dir2.glob('*') if item.name not in ['._.DS_Store', '.DS_Store', '._DS_Store']])

print (CLASS_NAMES)

test_data_gen1 = image_generator.flow_from_directory(directory=str(data_dir2), 
    batch_size=783, shuffle=True, target_size=(IMG_HEIGHT,IMG_WIDTH),
        classes=list(CLASS_NAMES))


CLASS_NAMES = np.array([item.name for item in data_dir3.glob('*') if item.name not in ['._.DS_Store', '.DS_Store', '._DS_Store']])

print (CLASS_NAMES)

test_data_gen2 = image_generator.flow_from_directory(directory=str(data_dir3), 
    batch_size=719, shuffle=True, target_size=(IMG_HEIGHT,IMG_WIDTH),
        classes=list(CLASS_NAMES))
```

An image set generation may be checked with s simple function that plots a subset of images. This is particularly useful when expanding the images using translations or rotations or other methods in the `image_generator` class, as one can view the images after modification.

```python
def show_batch(image_batch, label_batch):
    """Takes a set of images (image_batch) and an array of labels (label_batch)
    from the tensorflow preprocessing modules above as arguments and subsequently
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

### calls show_batch on a preprocessed dataset to view classified images

image_batch, label_batch = next(train_data_gen1)
show_batch(image_batch, label_batch)
```
Assigning the pair of 

```python
(x_train, y_train) = next(train_data_gen1)

(x_test1, y_test1) = next(test_data_gen1)

(x_test2, y_test2) = next(test_data_gen2)
```

Now comes the fun part: assigning a network architecture! The Keras `Sequential` model is a straightforward, if relatively limited, class that allows a sequential series of network architectures to be added into one model. This does not allow for branching architectures or other fancy neural network models, in which case the more flexible functional Keras module shouldb be used.

After some trial and error, the following architecture was found to be effective for the relatively noisy images I was classifying:

```python
model = tf.keras.models.Sequential([
    Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH , 3)),
    MaxPooling2D(),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(50, activation='relu'),
    Dense(2, activation='softmax')
])
```

Now the network can be compiled, trained, and evaluated on the test datasets. I also include `model.summary()` in order to have the specific of the model printed.

```python
model.compile(optimizer='Adam', 
	loss = 'categorical_crossentropy', 
	metrics=['accuracy'])

### Displays details of each layer output in the model 

model.summary()

### Trains the neural network, and print the progress at the end of each epoch
### (signified by verbose=2)

model.fit(x_train, y_train, epochs=9, batch_size = 20, verbose=2)

### Evaluates neural network on test datasets and print the results

model.evaluate(x_test1, y_test1, verbose=2)
model.evaluate(x_test2, y_test2, verbose=2)
```

