### Language Mixers III: Optimization

Training large models takes a large amount of compute, but why this is the case is not immediately apparent. On this page we take a first-principles approach to understanding why this is the case through the lens of numerical optimization.

### Background

The process of training the deep learning models that are used for language and vision modeling may be thought of as a specialized type of unconstrained nonlinear optimization, one in which the function or model being optimized is itself composed of many linear transformations separated by various normalizations and nonlinear functions, where each linear transformation and linear part of the normalization (but typically none of the nonlinear functions).

I think it is safe to say that if you dropped a 
