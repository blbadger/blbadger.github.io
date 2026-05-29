## Secrecy and Language Models

The question of whether or not two parties, one with an input and another with a model, can keep secrets from each other while effectively making use of that model is explored. Practical methods for performing language modeling such that the provider does not share most of their language modeling information and the user shares none of their input information are detailed.

### Introduction

It is common today (2026) for many individuals and organizations to send sensitive and otherwise confidential information to language models providers in order to enjoy the benefits of these models. The nature of language modeling means that these individuals and organizations must place a great deal of trust in the language model provider: although information may be encrypted during transit, the provider must be able to obtain an unencrypted form of the user's data in order to generate useful model outputs. There is nothing in principle from preventing these providers from obtaining the information fed to their models because of this.

A short while ago it was more or less inconcievable that such information would be shared in so unsecure a format. These observations motivate the central question of this work: can a language model user with an input keep the information content of that input confidential and still make use of the provider's model, while at the same time can the provider keep their model secret from the user?

The primary difficulty here lies in the necessity that the model gives a useful output to any given input. This precludes approaches to this problem where a user encrypts the input but does not decrypt before the language model observes the input, or where a user injects noise into an input to obfuscate it.

One approach to the problem of security and language model inputs is hardware encryption guarantees, which nvidia offers for newer datacenter GPUs. This may be considered to be an unsatisfactory solution to this problem for a number of reasons, but most of all because the user cannot actually verify that the ecryption is in place without accessing the provider's secrets (the model) and thus simply shifts the burden of trust onto a new element.

We exlore this problem with the assumption that the user and provider to undergo successful language modeling without some degree of cooperation, and focus on the particular case where a provider is willing to share some of their model's information with the user but the user seeks to minimize the information they send to the provider.

### Secrecy in LLMs

The question of how a provider and user would minimize the amount of information they would necessarily share with one another begins with an easier question: is it possible to perform language modeling in this scenario when the provider simply does not share all the model's information, and where the user likewise does not reveal all their information? Without further investigation the answer to this question can be determined as 'yes', as this follows from the results of [previous work](https://arxiv.org/abs/2602.13466) finding that language models are functionally non-invertible, meaning that one cannot train a model using a reasonable amount of compute to invert the last hidden layer's last token embedding to regenerate the input that was fed to the model to generate the embedding in the first place.

A system by which neither provider or user has access to all model and input information but always recieves the correct next token is as follows: first a provider shares all layers of their model except the language modeling head to the user, who then performs a forward pass on those layers to get the last token last hidden layer activations, and then sends those to the provider to recieve the next token. The results in the above paper imply that the provider cannot uniquely identify more that around 7\% of the tokens the embedding corresponds to, such that the exact information it contains remains secret and likewise the language modeling head layer remains secret from the user. 

This is notably not a very good secrecy system: with enough inputs the user will be able to closely approximate the provider's language modeling head transformation assuming that it is a single linear layer, and likewise although the provider cannot identify exact tokens the embedding still contains some useful input information. A natural question to ask is whether or not the provider could simply send fewer layers to the user and thus retain more model information without decreasing next token accuracy. 

This motivates the following question: what is necessary for language modeling secrecy in this sense?

### Theory: Secrecy and Invertibility

The two features of a model to fulfill perfect secrecy may be summarized as non-invertibility and good mixing.


### Practical Secrecy









