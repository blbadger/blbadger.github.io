### Introduction

This page is a continuation of Parts [I](https://blbadger.github.io/language-representations.html) and [II]((https://blbadger.github.io/language-representations-inputs.html)).  In [Part II](https://blbadger.github.io/language-representations-inputs.html) it was observed that hidden layer representations of the model input are typically hopelessly inaccurate, which is somewhat surprising being that vision transformers and convolutional models are capable of quite accurate input representation even in deeper layers.  This page begins by further testing representation accuracy before exploring the theory behind poor input representation for language models and concludes with a study on how representation can be more accurate and how how this accuracy affects language tasks.

### Restricted Vocabulary Input Representation

Why are trained GPT-2 models incapable of accurate input representation?  Observing the spatial pattern of feature and input representations (see [here](https://blbadger.github.io/language-representations.html) for work on this topic) gives us a hypothesis: in that work it is apparent that trained GPT-2 transformer blocks introduce high-frequency noise into the input representation of two images such that the argument maximum value of all pixels is no longer meaningful.  On this page we deal with the argmax of logits which are transformed via a Moore-Penrose pseudoinverse, but it stands to reason that a similar phenomenon is occuring here such that the argmax of the logits becomes unpredictable due to the introduction of high-frequency noise in $a_g$.

One way to test this idea is to note that the vast majority of the $50257$ tokens present in the GPT-2 vocabulary signify words or word parts that are extremely unlikely to be observed in real text.  Indeed it appears that these very tokens are the ones that tend to result when the $\mathrm{arg \; max}$ function is called on the logits of those token indicies for trained GPT-2. 

To prevent high-valued but rare tokens from being selected during the decoding process, we can simply mask all the tokens that are not selected (not in the `allowed_tokens` tensor) with 0 and proceed with argmax as before.

```python
def masked_decode(logits: torch.tensor, allowed_tokens: torch.tensor) -> str:
	mask_tensor = torch.zeros(logits.shape).to(device)
	mask_tensor[:, :, allowed_tokens] = 1.
	masked_logits = logits * mask_tensor
	allowed_output = torch.argmax(masked_logits, dim=2)[0]
	output = tokenizer.decode(allowed_output)

	return output
```

If we are very selective with our `allowed_tokens` and only allow words in the target input to be selected, we find that the input may be exactly recovered by a trained GPT-2 transformer block, such that the input representation is for $N=2000$ is $a_g = \mathrm{The \; sky \; is \; blue.}$  With some learning rate tuning and a sufficiently large $N$, we can recover the target input even for 4 transformer blocks, but for all 12 blocks it is rare that $a_g = a$ even after tuning such that $\vert \vert a_g - a \vert \vert < \vert \vert a' - a \vert \vert$.

This is remarkable in that even a very small expansion of the allowed vocabulary results in a trained GPT-2 block being unable to accurately recognize the input.  For example, given the words of the sentence 'The sky is blue or red depending on the time of day.' as our allowed tokens, after $N=20k$ steps we have 

$$
a_g = \mathtt{\;  day \; or \; day \; blue \; or}
$$

In contrast, the same model before training is capable of recovering the target input after $N=200$ steps. Note that neither trained nor untrained GPT-2 model with all 12 blocks is capable of recovering the input for even that slightly expanded vocabulary.

### Direct Input Representation

Early on this page, two methods to deal with the discrete nature of language inputs were noted: the first was to simply ignore them during the gradient descent process and convert the outputs back to tokens via the Moore-Penrose pseudoinverse, and this is arguably the most natural choice as it involves the fewest changes to the model itself.  Another option is to transform the input tokens (an array of integers) into a vector space and perform gradient descent directly with that space being the terminal node of the backpropegation computational graph.

To be precise, what we want is to convert an array of input tokens $a_t$, which for example could be

$$
a_t = [1, 0, 2]
$$

to continuous values in a vector space, which for the above example could be

$$
a = \begin{bmatrix}
0 & 1. & 0 \\
1. & 0 & 0 \\
0 & 0 & 1. \\
\end{bmatrix}
$$

but note that there is no specific rule for converting a token to a vector value, for instance we could instead assign any small value as a baseline with a larger value as the corresponding indicies.  For GPT-2, we can find what the trained embedding assigns for each input element by finding the Moore-Pensore pseudoinverse of the weight matrix of the embedding, denoted $W^+$, as follows:

$$
a = W^+E(a_t)
$$

where $a_t$ corresponds to an array of integers and $E$ signifies the embedding transformation that maps these integers to an embedding vector space. 

which can be implemented as 

```python
embedding = model.transformer.wte(tokens) 
embedding_weight = model.transformer.wte.weight.float() # convert to float in case model is in 16-bit precision
inverse_embedding = torch.linalg.pinv(embedding_weight)
logits = torch.matmul(embedding, inverse_embedding) # invert embedding transformations
```

Verifying that this procedure is accurate is not too hard, and may be done by first converting the vector space back to integer tokens via the $\mathrm{arg \; max}$ of each logit, and then decoding this array of integers as shown below.  

```python
tokens = torch.argmax(target_logits, dim=2)[0]
output = tokenizer.decode(tokens)
```

With this procedure, we find that the trained GPT-2 word-token embedding maps the value at the index of the integer token to a value on the order of $1 \times 10^{-2}$ whereas the values of other indices are typically $\leq 1 \times 10^{-3}$. We can also simply mandate that the vector values at indicies corresponding to the appropriate tokens have some large positive value and that all others are zero as follows:

```python
target_logits = torch.zeros(logits.shape).to(device)
for i in range(len(tokens[0])):
	target_logits[:, i, tokens[0, i]] = 10
```

There does not appear to be any significant difference between these two approaches to mapping $a_t \to a$.

Now that we have a continuous $a$ input, we can perform gradient descent on the model's output as follows:

$$
a_{n+1} = a_n + \eta * \nabla_{a_n} ||O_l(a_n, \theta) - O_l(a, \theta)||_1 \\
\tag{5}\label{eq5}
$$

where $a_0 = \mathcal N(a, \mu=1/2, \sigma=1/20)$.  The GPT-2 model also needs to be modified as the word-to-embedding transformation is designed to take in integer 'words' rather than vectors, and this can be ameliorated by replacing this transformation with a multiplication of the input by the word-to-embedding transformation weights as follows.

```python
class InputGPT(nn.Module):

	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, x: torch.tensor) -> torch.tensor:
		# replaces wte transformation
		x = torch.matmul(x, self.model.transformer.wte.weight)
  
		for i in range(12):
			x = self.model.transformer.h[i](x)[0]

		return x
```

With this direct gradient descent on the input method, can a single trained transformer block in GPT-2 be inverted accurately? Given the input 'This is a prompt sentence' the answer is no: iterating \eqref{eq5} such that $\vert \vert O_l(a_g, \theta) - O_l(a, \theta) \vert \vert < \vert \vert O_l(a', \theta) - O_l(a, \theta) \vert \vert$ (where $a'$ is a slightly shifted $a$ such that the tokenization of these vector values are identical) we have

```
 precarious NeighNASA Someonetre
lisherusersExp qualifying windshield
 ple propose18 joyful Genie
SHIP Geh lesbians inquiries Mat
1968 Carroll delibericycle consumers
```

Even when we limit the possible tokens to a very restricted set, specifically the tokens 

```
This is a prompt sentence with some extra words attached to increase the allowed vocabulary by a small margin
```

and optimizing \eqref{eq5} such that the inequality denoted above is observed, we have

$$
a_g =  \mathtt{This \; some \; by \; small \; is}
$$


indicating that a single trained GPT-2 transformer block is incapable of accurate input representation even for a restricted input vocabulary, even when the input is used to perform the gradient descent. This is also true when the gradient of \eqref{eq5} is calculated using the $L_2$ norm, indicating that it is not the choice of $L_1$ norm that prevents accurate input representation here.  

The same results are observed for one transformer embedding transformation and first block of the 1.5B parameter `gpt2-xl` model, such that neither the trained nor untrained model subsets are capable of accurate input representation even when the vocabulare is extremely limited.

For example, the top-5 decoding for the input 'This is a prompt sentence' for the word-to-embedding transformation followed by the first transformer block of an untrained `gpt2-xl` model is

```
 ple NeighNASA inquiries windshield
 duplusers 131 qualifyingtre
rypted Sharif deliber camping Genie
1968 Carroll maleicycle Humanity
 precariouscandExp semen consumers
 ```
 
 and for the trained model's embedding followed by the first transformer block, the top-5 input representations for the same prompt are
 
 ```
  silenced oppression Fav positioned�
 Shows debugprint interference Deploy
 softened949313 absent transports
 Ambrose mono unanswered chantingAM
ogacessive dungeonJR785
 ```

There is a clear explanation for why the GPT-2 embedding is non-invertible: the linear transformation corresponding to the token-to-embedding operation transforms a vector space of dimension $d(a) = 50257$ to a vector space of $d(O_e(a, \theta)) = 728$ for the base GPT-2 model, or $d(O_e(a, \theta)) = 1600$ for the 1.5B parameter `gpt2-xl`.  Non-invertibility is expected for both embeddings being that the output dimension is much smaller than the input, and as the input dimension exceeds the output there are many different inputs that will yield one identical output of the word to embedding transformation.

### Other attempts to enforce accuracy

Why do trained language models exhibit such poor input representation?  In the previous section, we found that the largest version of GPT2 exhibits far less repetition in its input representations than the smallest version of the same model.  Unfortunately it is also clear that the larger model is no more capable of producing a coherent input representation (even for one transformer block) even after 1000 gradient descent iterations, corresponding to a generated distance <1/6 the magnitude of the shifted distance.

It may also be wondered whether or not input representation would be improved if we used the output of multiple layers rather than only one. For the first three layers of a model in which we perform gradient descent on the input directly, this could be implemented as follows:

```python
class InputGPT(nn.Module):

	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, x: torch.tensor) -> torch.tensor:
		# replaces wte transformation
		x = torch.matmul(x, self.model.transformer.wte.weight)

		o1 = self.model.transformer.h[0](x)[0]
		o2 = self.model.transformer.h[0](o1)[0]
		o3 = self.model.transformer.h[0](o2)[0]
		output = torch.cat((o1, o2, o3))
		return output
```

But the input representations resulting from this are no better than before, and appears to confer the same ability to accurately represent an input as simply taking the output of the last (in this case the third) block.

Another possibility is that the metric we are using to perform gradient descent is not idea for input representation.  Specifically, instead of minimizing the $L^1$ metric

$$
m = || O(a, \theta) - O(a_g, \theta) ||_1
$$

we can instead minimize the cosine similarity (ie the cosine of the angle $\phi$) between vectorized (ie flattened) versions of the outputs $O(a, \theta)^*$

$$
\cos (\phi) = \frac{O(a, \theta)^* \cdot O(a_g, \theta)^* }{||O(a, \theta)^* ||_2 * |||O(a_g, \theta)^* ||_2}
$$

such that gradient descent on the first embedding $e$ is

$$
e_{n+1} = e_n - \nabla_{e_n} \cos (\phi)
$$

Performing gradient descent on the input of the first transformer block of a trained GPT-2 to minimize $\cos(\phi)$ turns out to lead to no more accurate input embeddings than before: reducing $\cos (\phi)$ to $0$ via 2000 iterations yields embeddings that may be inverted to give

```
Downloadha.""��MpServeriverpool
� httquerqueenglish Leilan
ARDISitudinalァidia guiActiveUn
�士)</etheless Vaugh��
?).andoエル millennOrderable
```

Even for a very limited vocabulary ('The sky is blue or red depending on the time of day.') the one transformer decoder module cannot accurately represent the input.

$$
\mathtt{depending \; depending \; time \; depending.}
$$

Therefore gradient descent is successful in minimizing the cosine distance between the output of the generated and target input (in this case embedding), but the generated input corresponds to nonsense.  The same is true even for single transformer decoders from untrained GPT-2 models, and as we earlier found that these modules can yield accurate input representations using gradient descent on the $L^1$ norm of the output difference, the cosine similarity may be viewed as a weaker measure for representation tasks.

Perhaps a linear combination (parametrized by constants $\alpha, \beta$ of the gradient of the cosine distance and a normed difference would be sufficient for accurate input representation from trained GPT-2 transformer modules.  

$$
e_{n+1} = e_n - \eta * \nabla_{e_n} \left( \alpha \cos (\phi) + \beta ||O(a_g, \theta) - O(a, \theta)||_1 \right)
$$

where $\eta$ is the learning rate hyperparameter, which is typically scheduled such that $\eta = 1 \to \eta = 1/100$ as the input representation iterations $n = 0 \to n = N$. But even for this metric the input representation after only one transformer encoder is quite inaccurate: for the first transformer block of an untrained GPT-2, we have input representations for 'The sky is blue.'

```
 execute228 therein blue.
 politicians jihadist18having Phill
 Given patriarchal Lightning Hughes sem
� discriminatelich antitrust fraternity
ucking demos� underrated310
```

which means that the linear combination is less accurate than using the $L^1$ norm loss alone.

### Implicit Language Tasks

It has been claimed that language models as they exist on this page are incapable of any kind of fact- or reason- based task because they are merely trained to predict the next word (or more accurately token) in some text. We can show that this claim is incorrect, however, using a fairly straightforward argument.  We can define a language model as some generalizable (to examples unseen in the training data) representation function that maps input token sequences to an output token such that the negative log-likelihood of that output token is minimized relative to the training dataset.  

From this definition alone we find that language models must also be capable of all the implicit tasks present in the training dataset.  For example, suppose the training data contained the phrase 'two plus two equals four'.  Given the input 'two plus two equals' the function must return 'four' if it has minimized the negative log-likelihood of that token. Now suppose the training data also contains the phrase 'four plus one equals five' and that now we give the unseen phrase 'two plus two plus one equals' and our function returns a token.  If our function is sufficiently generalizable, we are almost guaranteed to return 'five' because the internal representation of 'two plus two' is nearly equivalent to 'four' and the model has already learned that 'four plus one equals five'.

It has been empirically observed that models with more parameters are generally better at these implicit natural language tasks, which typically lie under the umbrella definition of 'reasoning' problems.  There are a few explanations for why this could be: firstly, larger models entail higher-dimensional space, such that gradient descent is more [biased towards generalization](https://arxiv.org/abs/2211.09639), secondly because more parameters imply greater 'memory' such that the model can learn more complicated representations of an input (ie 'three' being both a word and an implicit number) and thirdly because it is apparent that biological neural networks require a very large number of neurons to deal with language relative to other tasks, such as object recognition.

We can test how language models are capable of implicit reasoning tasks by observing the representations of various words in such models.

It may be wondered if even larger models (and those which are trained on far larger datasets) also suffer from the same tendency to for nonsensical representations of standard phrases. Given a sufficiently powerful representation method this would imply that even very large and powerful models are unable to tell much difference between meaningful sentences and gibberish.  We will examine `gpt-j` (which is a 6B parameter model trained on an 825 GB dataset), the `BLOOM-7b1` model with 7.1B parameters trained on a 1.4 TB dataset to find out.

...

If simply making a transformer-based language model larger and training it on more text is not able to lead to models becoming better able to represent their inputs, then what is? Language models today often follow general training in which the sole metric is predicting the next word in a sentence with what is termed 'aligmnent' which serves to make the language model return outputs that are aligned in some way to the task at hand (question answer, mathematics, etc.).  This alignment is usually achieved via supervised fine-tuning, deep reinforcement learning, or a combination of these two approaches.  

Does it matter for the purposes of language generation that even otherwise effective models are incapable of differentiating between nonsensical gibberish and meaningful sentences?  At first glance it may seem as though it may not matter: if a language model were to be given these nonsenical phrases then it may confuse them with actual text, but what are the chances that the exact nonsensical phrase would appear as a prompt to a language model in a real application?  

There is, however, reason to wonder whether it is not important that language models form such poor representations of their inputs.  Language models as they currently exist suffer from a significant and currently difficult-to-manage problem sometimes referred to as 'hallucinations', in which the model will return syntactically and semantically correct text that is woefully incorrect in the implicit language task at hand.  Furthermore, at present there appears to be no method that is capable of preventing this hallucination barring directly training against specific examples (either using supervised or reinforcement methods).  

This is fundamentally a problem of representation: if a language model were capable of representing all necessary implicit and explicit language tasks and inputs to a sufficient degree of accuracy, the model would be capable of discerning text that fails to address the implicit tasks from text that does not fail to do so.  As we have already seen that language models cannot represent their inputs uniquely, it may be little wonder why they are sometimes incapable of representing implicit input features as well.
