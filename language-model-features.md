## Language Model Features 

### Introduction

How do deep learning models make decisions?  This is a fundamental question for understanding the (often very useful) outputs that these models give.

One way to address this question is to observe that most deep learning models are compositions of linear operations (with nonlinear transformations typically applied element-wise) which means that one can attempt to examine each separable component of a model in relative isolation.  A particularly intuitive question one might ask is what input yields the maximum output for some hidden layer element (assuming other elements of that layer are relatively unchanged) because one may consider that input to be most 'recognizable' to that element.  In a numb

Feature visualization studies have shown that vision models learn to identify and generate shapes via a hierarchical sequence of shape speceficity model layer: early convolutional (or transformer) layers are most activated by simple patterns or repeated shapes whereas deeper layers are most activated by specific objects or scenes.  For much more information on this topic, see [this page](https://blbadger.github.io/feature-visualization.html).  

### Llama features explored


