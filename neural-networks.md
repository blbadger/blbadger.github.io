## Image classification with a [convolutional neural network](https://github.com/blbadger/neural-network).

### Preface: Neural networks are useful but not universal

This sort of qualifier is usually placed near the end of work on a program or type of program, but I think it is important enough to include before the main body of work.  All we need to know here is that neural networks are effective codes of finite length used for decision problems.

An oft-quoted feature of neural networks is that they are universal, meaning that they can compute any function (see [here](https://www.sciencedirect.com/science/article/abs/pii/0893608090900056) for a proof of this).  A good geometrical explanation of this is found in [Neilsen](http://neuralnetworksanddeeplearning.com/).  Being that many everyday tasks can be thought of as some kind of function, and  as neural networks are very good at a wide range of tasks (playing chess, recognizing images, language processing etc.) it can be tempting to see them as universal panacea for any problem.  This view is mistaken, however, and to see why it is best to understand what exactly universality entails before exploring the limits of any program.

There is an important qualification to the proof that neural networks can compute (more precisely, can arbitrarily approximate) any function: it only applies to continuous and differentiable functions.  This means that neural networks should be capable of approximating any smooth and connected function, at least in theory.  The first issue with this is whether or not a certain function is 'trainable', meaning whether or not a network is able to learn to approximate a function that is it theoretically capable of approximating.  Putting this aside, the need for differentiability and continuity brings a far larger qualification to the statement of universality.  

To see this, imagine that we were trying to use a neural network to approximate an arbitrary function.  What are the chances of this function being continuous and differentiable?  Let's see how many functions belong to one of three categories: differentiable (and continuous, as differentiability implies continuity), continuous but not necessarily differentiable, or not necessarily continuous or differentiable.  Visually, differentiable functions are smooth with no sharp edges or squiggles and continuous functions may have sharp edges but must be connected.  

Formally, we can define the set of all functions $f$ that map $X$ into $Y$, $f: X \to Y$:

$$
\{f \in (X, Y)\}
$$

with the set of all continuous functions being

$$
\{f \in \mathbf C(X, Y)\}
$$

and continuous, differentiable functions as

$$
\{f \in \mathbf C^1(X, Y)\}
$$

Using fundamental set theory, it can be shown that

$$
\lvert \{f \in \mathbf C^1(X, Y)\} \rvert << \lvert \{f \in \mathbf C(X, Y)\} \rvert << \lvert \{f \in (X, Y)\} \rvert
$$

in words, the size of the set of all continuous and differentiable functions is far smaller than the size of the set of all continuous functions, which is in turn far smaller than the set of all functions.  The usage of 'far smaller' does not quite do justice to the idea that each set is vanishingly smaller (really infinitely smaller) than the next.

What does this mean? If we restrict ourselves to differentiable and continuous function then neural networks are indeed universal, but they are hardly so for all possible functions.  

Could it be that a better neural network will be made in the future, and this will allow us to approximate nondifferentiable functions as well as differentiable ones?  Using more concrete terms, perhaps we will be able to engineer a neural network that is arbitrarily precise at decision problems (which is equivalent to classification) in the future.  This thinking can be used to represent the idea that although not perfect not, some sort of machine learning will be able to be arbitrarily good at decision making in the future.  If not neural networks then perhaps decision trees: can some sort of machine learning program get so good at classification that it can learn how to classify anything, to arbitrary precision?

The answer is no: no matter what program we use, we cannot solve most decision problems.  To see why, first note that any program, from a simple `print (Hello World!)` to a complicated neural network, is conveyed to a computer as a finite string of bits (a list of 1s and 0s).  Natural numbers are also defined by a finite string of bits and so we can establish an equivalence between finite programs and natural numbers.  The size of the set of all natural numbers is equivalent to the size of the set of all rational numbers (both are countably infinite)

$$
\{\mathtt {finite} \; \mathtt {programs}\} \approx \{finite \; strings\; of\; bits\} = \Bbb N 
$$

The size of the set of all natural numbers is equivalent to the size of the set of all rational numbers (both are countably infinite), so

$$
\{\mathtt {finite} \; \mathtt {programs}\} \approx \Bbb N \approx \Bbb Q
$$

Now let's examine the set of all possible decision problems.  We can restrict ourselves to binary classification without loss of generality, where we have two options: $1$ and $0$.  Now a binary decision problem has an ouput 1 or 0 for every input, and we can list the outputs as a string of bits in binary point, ie binary digits that define a number between 0 and 1.  As a decision problem must be defined for every possible input,  and as there are infinite inputs for any given decision problem, this string of bits is infinite.  


$$
\{decision\; problems\} \approx \{infinite \; strings\; of\; bits\} = \{x \in (0, 1]\}
$$


As the size of the set of all numbers in $(0, 1]$ is equivalent to the size of the set of all real numbers, 


$$
\{decision \; problems\} \approx \Bbb R
$$


and as the size of the set of all real numbers is uncountably infinite wheras the size of the set of all rational numbers is countably infinite,


$$
\{decision \; problems\} \approx \Bbb R >> \Bbb Q \approx \{\mathtt {finite} \; \mathtt {programs}\}
$$


This means that the set of all finite programs is a vanishingly small subset of the set of all decision problems, meaning that no finite program (or collection of programs) will ever be able to solve all decision problems, only an extremely small subset of them.

### Using a neural network to classify fluorescent images of cells

Perhaps image classification that can be done by humans falls into the category of problems that the neural network can at least partially solve. I have taken many images of biological specimens (usually fluorescence images of cells) over the last decade, and think I have gotten pretty good at determining which images are of what category.  Can an automated system perform this task as good or better than me?

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

The input shape must match the dataset image size, and the output must be the same number of possible classifications.  In this case, we are performing a binary classification between two cells, so the output layer of the neural network has 2 neurons.  After some trial and error, the following architecture was found to be effective for the relatively noisy images I was classifying:

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
    Dense(2, activation='softmax')
])
```
Which may be represented graphically as

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/neural_network.png)

For optimal test classification accuracy at the expense of longer training times, an extra dense layer of 50 neurons was added as follows:

```python
...
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

### The network mimics expert human image classification capability

Using blinded approach, I classified 93 % of images correctly for certain test dataset, which we can call 'Snap29' after the gene name of the protein that is depleted in the cells of half the dataset (termed 'Snap29') along with cells that do not have the protein depleted ('Control').  There is a fairly consistent pattern in these images that differentiates 'Control' from 'Snap29' images: depletion leads to the formation globs of fluorescent protein in 'Snap29' cells.

The network shown above averaged ~96 % accuracy (over a dozen training runs) for this dataset.  We can see these test images along with their predicted classification ('Control' or 'Snap29'), the confidence the trained network ascribes to each prediction, and the correct or incorrect predictions labelled green or red, respectively.  The confidence of assignment is the same as the activation of the neuron in the final layer representing each possibility.  This can be achieved as follows:

```python
### Creates a panel of images classified by the trained neural network.

image_batch, label_batch = next(test_data_gen1)

test_images, test_labels = image_batch, label_batch

predictions = model.predict(test_images)

def plot_image(i, predictions, true_label, img):
    """ returns a test image with a predicted class, prediction
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

### I am not sure why the num_cols and num_rows do not add up to the total figsize, 
### but keep this in mind when planning for a new figure size and shape

num_rows = 4
num_cols = 3
num_images = 24

### Plot initialization

plt.figure(figsize = (num_rows, num_cols))

### Plot assembly and display

for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, i+1)
  plot_image(i+1, predictions, test_labels, test_images)

plt.show() 
```

which yields

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/nn_images_1.png)


Let's see what happens when the network is applied to an image set without a clear difference between the 'Control' and experimental group (this time 'Snf7' after the protein depleted in this instance).  After being blinded to the true classification labels, I correctly classified 70.7 % of images of this dataset.  This is better than chance (50 % classification accuracy being binary) but much worse than for the Snap29 dataset. Can the neural network do better?

It cannot: the average training run results in 62 % classification accuracy, and the maximum accuracy achieved was 66 %, both slightly lower than my manual classification accuracy.  We can see the results of one particular training run: the network confidently predicts the classification of nearly all images, but despite this confidence is incorrect on the identity of many.

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/nn_images_2.png)

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

Thus the network overfits (memorizes) Snf7 images but is able to effectively differentiate Snap29 images.  This makes sense from the perspective of manual classification as there is a relatively clear pattern that one can use to distinguish Snap29 images, but little pattern to identify Snf7 images.  

A slower learning process for generalizable learning is observed compared to that which led to overfitting, but perhaps some other feature could be causing this delay in training accuracy.  The datasets used are noticeably different: Snap29 is in two colors whereas Snf7 is monocolor.  If a deeper network (with the extra layer of 50 dense neurons before the output layer) is trained on a monocolor version of the Snap29 or Snf7 datasets, the test accuracy achieved is nearly identical to what was found before,

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/nn_images_5.png)

And we see that this deeper network continues to confidently predict incorrect classifications for Snf7

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/nn_images_6.png)

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/nn_images_7.png)

Once again Snap29 training accuracy lags behind that of Snf7.

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/nn_images_8.png)

Even when the Snap29 and Snf7 datasets are matched in size, the sharp increase in training accuracy seen for Snf7 is seen once again, suggesting that this effect does not result from differences in dataset size.

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/nn_images_9.png)

To see if the faster increase in training accuracy for Snf7 was peculiar to the particular network architecture I used, I designed a network that mimics the groundbreaking AlexNet, and the code to do this may be found [here](https://github.com/blbadger/neural-network/blob/master/AlexNet_sequential.py).  Using this network, it has previously been seen that overfitting is the result of slower increases in training accuracy relative to general learning (see [here](https://arxiv.org/abs/1611.03530) and [here](https://dl.acm.org/doi/10.5555/3305381.3305406)).  With the AlexNet mimic, once again the training accuracies for Snf7 increased faster than for Snap29 (although test accuracy was poor for both datasets).  This suggests that the faster training leading to overfitting in the Snf7 dataset is not peculiar to one particular network architecture and hyperparameter choice.

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/nn_images_10.png)

![neural network architecture]({{https://blbadger.github.io}}/neural_networks/nn_images_11.png)


### Learning is not global minimization of the cost function

Neural networks learn by adjusting the weights and biases of the neurons in order to miminize a cost function, which is a continuous and differentiable representation of the accuracy of the output of the network. This is usually done with a variant on stochastic gradient descent, which involves calculating the gradient (direction of largest change) using a randomly chosen subset of images (which is why it is 'stochastic').  This is conceptually similar to rolling a ball representing the network down a surface representing the multidimensional weights and biases of the neurons.  The height of the ball represents the cost function, so the goal is to roll it to the lowest spot during training.  

As we have seen in the case for Snf7 dataset images, this conception is accurate for increases in training accuracy but not not necessarily for general learning: reaching a global minimum for the cost function (100 % accuracy, corresponding to a cost function of less than 1E-4) led to severe overfitting and confident prediction of incorrect classifications.  This phenomenon of overfitting stemming from a global minimum in the cost function during training is not peculiar to this dataset either, as it has also been observed [elsewhere](http://proceedings.mlr.press/v38/choromanska15.pdf).  

If decreasing the neural network cost function is the goal of training, why would an ideal cost function decrease (to a global minimum) not be desirable?  In our analogy of a ball rolling down the hill, something important is left out: the landscape changes after every minibatch (more accurately, after every computation of gradient descent and change to neuronal weights and biases using backpropegation).  Thus as the ball rolls, the landscape changes, and this change depends on where the ball rolls. 


### Neural networks are systems of dimension reduction: implications for the presence of adversarial examples

Neural networks, like any statistical learning procedure, are in the business of dimensional reduction.  This is because they take in inputs that are necessarily larger than outputs, which may seem counterintuitive if the inputs are small images and the outputs are choices between thousands of options.  Even then, dimensional reduction holds: to see this, suppose that each image were classified into its own category.  Then the network would not reduce dimension but the classification would be trivial: any program could do just as well by classifying any image to its own category.  In the process of assigning multiple inputs to the same category, dimensional reduction occurs.

As seen for the nonlinear attractors [here](\clifford_attractor.md), changes in dimension are rarely smooth: small changes in inputs lead to large changes in attractor shape. This is important because it also applies to neural networds, and is evidenced by the existence of [adversarial negatives](https://arxiv.org/abs/1312.6199), images that are by eye indistinguishable from each other but are seen by a network to be completely different.  











