Structured Repeat Mixers

THe most efficient non-trivial language model for autoregressive inference are models that read the current token and a fixed memory and predict a next token.  This is how recurrent neural network perform inference, but training these models efficiently usually requires a reformulation for efficient training. One way to do this is to 

On this page we explore the adapation of masked mixers to linear time and constant space complexity language modeling. These models

### Background and Hypotheses



Thus we have our first main hypothesis: adapting mixer layers for linear time and constant space complexity sequence modeling should be much easier than doing so for self-attention or state spaces. 

### R

### What Token Mixing Weights do Masked Mixers Learn?


