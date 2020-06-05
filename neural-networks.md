## Image classification with a convolutional neural network

Say that you were tasked with classifying images using an automated system, one that does not depend on user input.  There could be many reasons for this, and the one that motivated the making of this particular network was an effort to assess reproducibility.  I have taken many images of biological specimens (usually fluorescence images of cells) over the last decade, and think I have gotten pretty good at determining which images are of what category.  Can an automated system perform this task as good or better than me?

Programs that classify complicated things often fall into the category of machine learning.  Machine learning is a subset of supervised statistical learning techniques, some of which may be more familiar than others.  Statistical learning is the process of pattern identification in statistical data.  Linear regression, decision trees, and neural networks are all supervised statistical learning techniques, meaning that these techniques use information from a training dataset to predict the outputs of a test dataset.  

Supervised techniques are in the business of reproduction: in the case of image classification, we already 'know' how to classify a certain number of images that becomes our training dataset, and we seek to classify other images that the program has not seen before.  In order to assess accuracy for classification on this test dataset, we need to know the true classification to be able to compare this with the predicted classification.  The ideal supervised learning technique takes a training dataset and accurately generalizes from this such that unseen datasets are classified correctly using information from this training dataset. In essence, we are seeking to reproduce the previous classification of images in the training dataset.

The current state-of-the-art method for classifying images is with neural networks.  These are programs with nodes called 'neurons' that change in accordance to the training dataset such that classification of test data is achieved. 

Perhaps the best introduction to neural networks is an online book by Neilsen found [here](http://neuralnetworksanddeeplearning.com/).  This book shows one how to set up a neural network from scratch in order to best understand what these programs to.  For speed and ease of modification, there are a number of libraries that exist for the purpose of making neural networks: Theano, Keras, Tensorflow, and PyTorch are a few for python.  I decided to try Tensorflow (which uses a Keras front end) and found a number of introductory materials extremely useful in setting up a neural network, perhaps the most accessible being the official [Keras introduction](https://www.tensorflow.org/tutorials/keras/classification) and [load image](https://www.tensorflow.org/tutorials/load_data/images?hl=TR) tutorials. 









