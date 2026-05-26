## Secrecy and Language Models

The question of whether or not two parties, one with an input and another with a model, can keep secrets from each other while effectively making use of that model is explored here. 

### Introduction

It is common today (2026) for many individuals and organizations to send sensitive and otherwise confidential information to language models providers in order to enjoy the benefits of these models. The nature of language modeling means that these individuals and organizations must place a great deal of trust in the language model provider: although information may be encrypted during transit, the provider must be able to obtain an unencrypted form of the user's data in order to generate useful model outputs. There is nothing in principle from preventing these providers from obtaining the information fed to their models because of this, and a short while ago it was more or less inconcievable that such information would be shared in so unsecure a format. These observations motivate the central question of this work: can a language model user with an input keep the information content of that input confidential and still make use of the provider's model, while at the same time can the provider keep their model secret from the user?

A current approach to the problem of security and language model inputs is hardware encryption guarantees, which nvidia offers for newer datacenter GPUs. This may be considered to be an unsatisfactory solution to this problem for a number of reasons, but most of all because the user cannot actually verify that the ecryption is in place without accessing the provider's secrets (the model) and thus simply shifts the burden of trust onto a new element.

It is clearly difficult for user and provider to undergo successful language modeling without some degree of cooperation.

### Secrecy Systems and Invertibility
