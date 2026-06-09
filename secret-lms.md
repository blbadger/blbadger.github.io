## Secrecy and Language Models

The question of whether or not two parties, one with an input and another with a model, can keep secrets from each other while effectively making use of that model is explored. Practical methods for performing language modeling such that the provider does not share most of their language modeling information and the user shares none of their input information are detailed.

### Introduction

It is common today (2026) for many individuals and organizations to send sensitive and otherwise confidential information to language models providers in order to enjoy the benefits of these models. The nature of language modeling means that these individuals and organizations must place a great deal of trust in the language model provider: although information may be encrypted during transit, the provider must be able to obtain an unencrypted form of the user's data in order to generate useful model outputs. There is nothing in principle from preventing these providers from obtaining the information fed to their models because of this. A short while ago it was more or less inconcievable that such information would be shared in so unsecure a format. These observations motivate the central question of this work: can a language model user with an input keep the information content of that input confidential and still make use of the provider's model, while at the same time can the provider keep their model secret from the user?

The primary difficulty here lies in the necessity that the model gives a useful output to any given input. This precludes approaches to this problem where a user encrypts the input but does not decrypt before the language model observes the input, or where a user injects noise into an input to obfuscate it. It is trivial to give a secrecy system that simply corrupts or removes sensitive input information, but as this clearly changes the behavior of the model's output it is undesirable to do so.

One approach to the problem of security and language model inputs is hardware encryption guarantees, which for example nvidia offers for newer datacenter GPUs. This may be considered to be an unsatisfactory solution to this problem for a number of reasons, but most of all because the user cannot actually verify that the ecryption is in place without accessing the provider's secrets (the model) and thus simply shifts the burden of trust onto yet another element that is controlled by the provider.

We exlore this problem with the assumption that the user and provider to undergo successful language modeling without some degree of cooperation, and focus on the particular case where a provider is willing to share some of their model's information with the user but the user seeks to minimize the information they send to the provider.

### Secrecy in LLMs

The question of how a provider and user would minimize the amount of information they would necessarily share with one another begins with an easier question: is it possible to perform language modeling in this scenario when the provider simply does not share all the model's information, and where the user likewise does not reveal all their information? Without further investigation the answer to this question can be determined as 'yes', as this follows from the results of [previous work](https://arxiv.org/abs/2602.13466) finding that language models are functionally non-invertible, meaning that one cannot train a model using a reasonable amount of compute to invert the last hidden layer's last token embedding to regenerate the input that was fed to the model to generate the embedding in the first place.

A system by which neither provider or user has access to all model and input information but always recieves the correct next token is as follows: first a provider shares all layers of their model except the language modeling head to the user, who then performs a forward pass on those layers to get the last token last hidden layer activations, and then sends those to the provider to recieve the next token. The results in the above paper imply that the provider cannot uniquely identify more that around 7\% of the tokens the embedding corresponds to, such that the exact information it contains remains secret and likewise the language modeling head layer remains secret from the user. 

This is notably not a very good secrecy system: with enough inputs the user will be able to closely approximate the provider's language modeling head transformation assuming that it is a single linear layer, and likewise although the provider cannot identify exact tokens the embedding still contains some useful input information. A natural question to ask is whether or not the provider could simply send fewer layers to the user and thus retain more model information without decreasing next token accuracy. 

This motivates the following question: what is necessary for language modeling secrecy in this scenario?

### Theory: Secrecy and Invertibility

In the context of secrecy models, perfect secrecy requires that the model be expressed as a non-invertibile (more precisely non-injective) function that mixes a sufficiently large input space. We examine the first quality before proceeding to the second. A non-injective is one in which maps many distinct inputs to one single output. As currently constructed, language models are highly non-invertible (composed of many layers of non-invertible transfomrations) and fulfill this criteria almost trivially, but in the sense of next token prediction these models are also funcitonally non-invertible, as ealier mentioned, because one cannot typically regenerate the input sequence of tokens given a vector sufficient to map to the output (the last hidden layer of the last token). The likelihood of invertibility in this functional sense drops precipitously as the number of tokens in th einput sequence increases as the relative amount of information present in the last token's last hidden layer decreases relative to the input's total information.

As we shall see on this page, however, although strictly non-invertible next token prediction language models are functionally invertible if hidden layers from *all* tokens are supplied to a decoder. As a full-input embedding must be given for the provider to keep secret more than just the language modeling head transformation, this paradigm is particularly important for the following discussion of applications.

### Perfect Secrecy

After [Shannon](https://pages.cs.wisc.edu/~rist/642-spring-2014/shannon-secrecy.pdf) first consider the case of perfect secrecy, defined as where the probability distribution of a message over all potential messages is unchanged after one recieves an encryption of that message. We ignore the information yielded by the model's prediction of the next token, as for certain architectures the provider would not have this information either.

In the classical sense, a message can be encrypted using a key at least as large as the message itself such that the number of encryptions is at least as large as the number of messages in order to provide perfect secrecy where the probability that a message has identity $M$ is unchanged if we are given the encoding of that message, $E$, or in symbols $P(M) = P_E(M)$ and which by Bayes theorem is equal to $P(E) = P_M(E)$. An example of perfect secrecy where $\vert M \vert = \vert E \vert = n$ was given by Shannon as follows: for encryption method $T$ mapping messages $M$ to encodings $E$, where $n$ messages are indexed $M_j \in \{ M_0, M_1, ..., M_n \}$ and similarly $E_s \in \{ E_0, E_1, ..., E_n \}$ and $T_i \in \{ T_0, T_1, ..., T_n \}$ we then have

$$
T_iM_j = E_s
$$

with $s = i + j \pmod  n$, this results in $P(E) = P_M(E) = 1/n$ fulfilling the condition of perfect secrecy.

We must adapt this theory to use with the language modeling scenario defined above because ciphering via $T$ must be restricted to generate encodings $E_s$ that are themselves useful natural language token sequences. We define a `useful' encoding as one that yields the same next token (or next token probability distribution for sampled models) when fed to a language model as the original message $M_j$. The language model $\theta$ performs a transformation of potential input sequences $a$ to a single output token $b$, denoted as $b = O(a, \theta)$, which in the context of a perfect secrecy system can be represented as follows:

$$
O(T_iM_j, \theta) = O(E_s, \theta)
$$

It is almost always safe to say that many distinct $b$ are mapped to one $a$ (which is certainly the case for natural language) such that $O: a \to b$ performs a non-invertible mapping. To create a perfect secrecy system, we proceed as follows: first for any given message $m$ we assemble a (potentially infinite) set of equivalent messages $M_j$ such that for all $k, l$ we have $O(M_k, \theta) = O(M_l, \theta), k \neq l$, then we map to an encoding via $T_i: M_j \to E_s$ ($E_s$ is also an element of $M$ by definition) to receive our encoding, which yields the same output when fed to the provider's model but reveals nothing about the actual input sequence. Conceptually this procedure may be stated as follows: if we can map our secret message to the set of all messages that yield the same next token when fed to a provider's model, we can simply swap our phrase for a randomly chosen element of this set and reveal nothing about our message assuming that the set is very large (typically it is infinitely large ignoring context window limitations). The two necessary elements for this procedure are non-invertibility of the model (so that the set $M$ is larger than 1) and input mixing (so that $m$ can be swapped with $M_j$). 

There is a substantial practical problem with this approach, however: if we were to find a function $F$ to generate the set $M$ that results in the same next token as our message $m$, we end up with a function that infers the same next token as the provider's model. This means that we would not actually need to use the provider's model at all, and thus we turn to practical secrecy methods that do not require a recapitulation of the provider's model.

### Practical Secrecy

Practical secrecy for the user/provider language modeling paradigm may be defined as follows: can the user and provider share minimal information with each other while undergoing successful modeling, where the provider cannot realistically be expected to recover the user's information given the compute they may have access to? 

Before directly addressing this question, we can answer a simpler one: assuming that the provider has no compute or any other codebreaking method, can user and provider exchange minimal information and succeed in their modeling? The answer is yes, and one method that fulfills this criteria is as follows: first the provider sends the user a certain number of layers, say 1/4 or 1/2, of their model, secondly the user performs gradient descent on an initially random input in order to match the output of the last layer sent to the output of their secret message, and then they send this transformed input (actually the embedding of this input) to the provider instead of their message. [Previous work](https://blbadger.github.io/language-representations-inputs.html) has shown that such generated embeddings essentially never match the input that one uses to generate the target output as long as the target is not in the first few model layers. The input generation process is conditioned on a random starting point that depends on the seed one uses, such that for any message there are many (infinite) generated inputs that all yield the correct next token.

For an example, suppose we had the following secret message:

**This is a secret message, not to be shared with anyone ever. The contents of this message are so obfuscated, so unknowable, that no one will ever be able to find what they are. The message is: The true identity of Satoshi Nakamoto is Spongebob Squarepants. End Message.**

for a small 16-layer transformer model trained for next token prediction on FineWeb, if we perform this input generation procedure with three different random seeds (random initial states) we generate embeddings that map to the following tokens:

```markdown
sign所所Batelizeomanip welt摄bebby Sob.ăng bby ofainathiselize inopleabweanik andOf of crest andeach.ofchina服obleoot Caldwellbyculo liesbybyAppearbyossal服ieuxof/original ofelize_ABI район/masterhaltainaainaoleonferenceselizeampa娘elize
```

```markdown
sign所所 Carryelize(ns welt spiralRVby Sobelixăng bby ofainathiselize in висabweanik andOf of and andeach.of Zukoot visitorongsTo(nsbyculo易bybyAppearbyoyal服ieuxờiifth ofelizearchyspath/masterhaltainailtonoleonendoza"},ampa娘 cue
```

```markdown
所`.`elizeapiro welt kRVяти Sob.ăng belize ofainathiselize in_soabwein andOf ofяб andeach.of ZukIobleongsTo Caldwellbyculo is andbyisetbyoyalidgeieuxofifth ofelizearchyspath khaltainailtonoleonferenceselizeampa娘 b
```

which are clearly distinct although they do contain a somewhat similar subset of input tokens, and in no way resemble the secret message. 

Now the more difficult question: can user and provider exchange minimal information for successful modeling assuming that the provider attempts to recover the user's input information? In an earlier section we saw if the provider is willing to share nearly all of the model with the user, and the user accepts that the provider will be able to identify around 7% of their input tokens, then the answer is yes. But it is unlikely that a provider would consent to share nearly all of their model with the user as is necessary in that method, nor is it likely that a user would be happy with only around 93% secrecy.

The difficulty here is that if the provider wishes to withold most of their model, and if we assume the provider uses a transformer model, the user must supply not just the last token's last hidden layer but all token's nth hidden layer embeddings to the provider. For causal transformers doing so results in a practically invertible system: we can train a decoder to take the output of all tokens of the user's portion of the model and regenerate the input sequence, which is notably not the case if a single token's embedding is used.

It turns out that if the provider expends some compute and effort to decipher the obfuscated inputs given by the gradient descent method above, they can determine the original message without too much trouble. The intuition here is that although many inputs map to one output, the inputs generated above are never actually found in the training dataset and thus a trained model can simply map these back to the corresponding real inputs. A decoder trained to invert a language model's encoding turns out to be sufficient to decode these obfuscated inputs.

### Secrecy with Current Architectures

The structure of LLM architectures today is remarkably homogenous: practically every large model consists of a sequence of modues, each composed of a token mixing layer (usually self-attention or hybrid attention-state space) and a feedforward layer on each token. To simplify this discussion, we refer to the output activations of these modules as 'layers'. Architectural details are not particularly important for this discussion aside from the sequential nature of models, where the knowledge of one hidden layer (for all tokens) is sufficient to complete the forward pass and get a next token output. This means that the user can retain any first n layers to keep information from the provider, but retaining the last n layers cannot possibly keep information from the provider because they will always be able to simply complete the forward pass. 

Arguably the most important effect of the provider always being capable of obtaining know the identity of any output token (if they retain any part of the model at all) is that the provider can trivially assemble a list of non-encoded output tokens for each prompt. This effectively makes KV caching not useful for long conversations, in the sense that the provider is more and more likely to oncover the user's secrets simply by observing the output tokens produced. The user can circumvent this issue by maintaining many conversations and swapping encoding methods for each next token for each conversation, as then the provider has no knowledge of which conversation corresponds to which input without being able to decode the input. This is notably not the case if KV caching is used, however, as the provider can simply reference the KV vectors they retain to know the identity of the conversation (in terms of output tokens).

Nevertheless, it can be shown that even using sequential architectures can result in perfect (even if impracticaly without KV cache ability) secrecy, and how this can be done is as follows:

First consider an arbitrary sequential model trained for next token prediction, which we call $P$, composed of L layers in total. To share some but not all (or even most) of this model's information with the user, the provider can send them a certain number of layers starting with the token embedding transformation, which we can think of as an encoder $E = P_{0:n-1}$ while retaining the rest of the layers as a causal decoder $D = P_{n:L}$. In this paradigm, the user takes their message $M$ and encodes it via applying the layers they recieve from the provider to make $e = E(M) \in \Bbb R^{cd}$, where $c$ signifies the context size in terms of the number of tokens $n_{ctx}$ and $d$ the hidden layer dimension, and then sends this encoding to the provider who completes the forward pass and provides the next token to the user.

The encoding $e$ is not strictly speaking in the clear in the sense that one would be able to recover $m$ with no effort, but it is also not a very good encoding for most model types because 

### Secrecy with Any Architecture

In the last section we considered sequential models and showed how one can perform secrecy obfuscation using combinations of secret encoders. The primary disadvantage of such efforts is that 1) the provider will still be able to obtain the next token, and because of this 2) the secret encoder training method is involved, requiring many models to be trained and utilized.

Happily both of these are features of sequential models rather than language modeling in general. To show that this is the case, consider a counterexample in which a model had a sequential stack of layers similar to current transformers, but a parallel stack (say of many fewer layers) that took as inputs the output of the first layer of the first stack, and gave an output to the input of the last layer. This input can be as simple as a linear combination between layers or else a more complicated operation. In effect this is a model with both sequential and parallel modules, architectures which proved very effective for image modeling in the hands of Google (see GoogleNet). 

It is apparent that a provider that retains many layers from such a model typically does not know the identity of the output token, as the output depends on both sequential stacks and the provider may retain only one. This means that the provider does not have knowledge of the user's next token upon each forward pass, which has the notable advantage of allowing for KV caching to greatly speed up inference. 

This property of not knowing an output makes the training of secrecy encoders much simpler too. 


### Applications

Modern encryption typically falls short of perfect secrecy as defined in the last section because one usually seeks to use a smaller encryption cipher than the message. 







