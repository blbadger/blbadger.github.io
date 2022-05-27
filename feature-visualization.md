## Feature Visualization and Deep Dreams

### Introduction

Suppose one wished to perform a very difficult task, but did not know how to begin.  In mathematical problem solving, one of the strategies most commonly used to approach a seemingly impossible problem is to convert that problem into an easier one, and solve that.  The principle 'make it easier' is considered by some mathematical competition instructors (Zeitz etc.) as being perhaps the most important of all problems.

In many respects, the field of deep learning places the same importance on the 'make it easier' principle.  Most problems faced by the field certainly fall into the category of being very difficult: object recognition, indeed, is so difficult that no one has made much progress in this problem by directly programming a computer with explicit instructions in decades.  The mathematical function mapping an input image to its corresponding output classification is therefore very difficult to find indeed.

Classical machine learning uses a combination of ideas from statistics, information theory, and computer science to try to find this and other difficult functions in a limited number of steps (usually one). Deep learning distinguishes itself by attempting to approximate functions using more than one step, such that the input is represented in a better and better way before a final classification is made.  In neural networks and similar deep learning approaches, this representation occurs in layers of nodes (usually called neurons).  


![adversarial example]({{https://blbadger.github.io}}/neural_networks/inceptionv3_architecture.png)

(image source: Google [docs](https://cloud.google.com/tpu/docs/inception-v3-advanced))

### Feature Visualization 

The process of sequential input representation, therefore, is perhaps the defining feature of deep learning.  This representation does make for one significant difficulty, however: how does one know how the problem was solved?  We can understand how the last representation was used to make the output in a similar way to how other machine learning techniques give an output from an input, but what about how the actual input is used?  The representations may differ significantly from the input itself, making the general answer to this question very difficult.

One way to address this question is to try to figure out which elements of the input contribute more or less to the output.  This is called input attribution.

Another way is to try to understand what the representations of each layer actually respond to, and then try to understand how this response occurs.


