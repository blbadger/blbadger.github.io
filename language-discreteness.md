## Language Modeling and Discrete Encodings

This page is a continuation of Parts [I](https://blbadger.github.io/language-representations.html) and [II](https://blbadger.github.io/language-representations-inputs.html). 

### Introduction

In [Part II](https://blbadger.github.io/language-representations-inputs.html) it was observed that hidden layer representations of the model input are typically hopelessly inaccurate, which is somewhat surprising being that vision transformers and convolutional models are capable of quite accurate input representation even in deeper layers.  This page begins by further testing representation accuracy before exploring the theory behind input representation accuracy for language models, and theory as to why this accuracy is important to language modeling tasks.

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

For clarity, the following figure provides a summary of the direct input representation method.

![direct input representation]({{https://blbadger.github.io}}deep-learning/llm_direct_representation.png)

With this direct gradient descent method, can a single trained transformer block in GPT-2 be inverted accurately? Given the input 

$$
a = \mathtt{This \; is \; a \; prompt \; sentence}
$$

the answer is no: iterating \eqref{eq5} such that $\vert \vert O_l(a_g, \theta) - O_l(a, \theta) \vert \vert < \vert \vert O_l(a', \theta) - O_l(a, \theta) \vert \vert$ (where $a'$ is a slightly shifted $a$ such that the tokenization of these vector values are identical) we have

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

There is a clear explanation for why the GPT-2 embedding transformation is non-invertible: the linear transformation corresponding to the token-to-embedding operation transforms a vector space of dimension $d(a) = 50257$ to a vector space of $d(O_e(a, \theta)) = 728$ for the base GPT-2 model, or $d(O_e(a, \theta)) = 1600$ for the 1.5B parameter `gpt2-xl`.  Non-invertibility is expected for both embeddings being that the output dimension is much smaller than the input, and as the input dimension exceeds the output there are many different inputs that will yield one identical output of the word to embedding transformation.  

But [elsewhere](https://blbadger.github.io/language-representations-inputs.html#sentence-representation-with-language-models) we have seen that non-invertible transformations can be effectively inverted using gradient descent, in the sense that an accurate input representation can be made accross non-invertible transformations.  Thus it remains to be seen whether direct input representations can be accurate for other models, even if they are not accurate for GPT-2.

### Multiple hidden layer outputs do not improve input representation

Why do trained language models exhibit such poor input representations?  In the previous section, we found that the largest version of GPT2 exhibits far less repetition in its input representations than the smallest version of the same model.  Unfortunately it is also clear that the larger model is no more capable of producing a coherent input representation (even for one transformer block) even after 1000 gradient descent iterations, corresponding to a generated distance <1/6 the magnitude of the shifted distance.

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

### Indirect input representation via cosine similarity

Another possibility is that the metric we are using to perform gradient descent is not idea for input representation.  Specifically, instead of minimizing the $L^1$ metric

$$
m = || O(a, \theta) - O(a_g, \theta) ||_1
$$

we can instead maximize the cosine similarity (ie the cosine of the angle $\phi$) between vectorized (ie flattened) versions of the output of the target input's embedding $O(e, \theta)^* $ and the vectorized output of the generated input embedding at iteration $n$, $O(e_n, \theta)^*$

$$
\cos (\phi) = \frac{O(e, \theta)^* \cdot O(e_n, \theta)^* }{||O(e, \theta)^* ||_2 * ||O(e_n, \theta)^* ||_2}
$$

such that gradient descent on the embedding $e_n$ is performed as follows:

$$
e_{n+1} = e_n - \eta \nabla_{e_n} \left( 1 - \cos (\phi) \right)
$$

where $\eta$ is a tunable learning rate parameter. Performing gradient descent on the input of the first transformer block of a trained GPT-2 to minimize $\cos(\phi)$ turns out to lead to no more accurate input embeddings than before: increasing $\cos (\phi)$ to $0.96$ via 500 iterations yields embeddings that may be inverted to give

```
[] become Holiday trollingMa calls
 UnierierylCAP PearlOTA
Hard obviousbys abiding�士73
arcer underCoolmanship autobiography outcomes
 slog bringsquaDirgap {:
```

Even for a very limited vocabulary ('The sky is blue or red depending on the time of day.') the one transformer decoder module from GPT-2 cannot accurately represent the input, although a significant improvement is made.

$$
\mathtt{This \; attached \; with \; prompt \; sentenceThis.}
$$

Therefore gradient descent is successful in minimizing the cosine distance between the output of the generated and target input (in this case embedding), but the generated input corresponds to nonsense.  The same is true even for single transformer decoders from untrained GPT-2 models, and as we earlier found that these modules can yield accurate input representations using gradient descent on the $L^1$ norm of the output difference, the cosine similarity may be viewed as a weaker measure for representation tasks.

Perhaps a linear combination (parametrized by constants $\alpha, \beta$ of the gradient of the cosine distance and a normed difference would be sufficient for accurate input representation from trained GPT-2 transformer modules.  Specifically we can see if the following update on the embedding is capable of accurate input representation. The update we make is

$$
e_{n+1} = e_n - \eta * \nabla_{e_n} \left( \alpha (1 - \cos (\phi)) + \beta ||O(a_g, \theta) - O(a, \theta)||_1 \right)
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

### Model Interpretability

Given the extreme difficulty in accurate input representation typical of large language models when using gradient descent on an initially random input, it may be wondered whether the gradient on the input is capable of offering any useful information.  This may be tested in a number of different ways, but one of the simplest is to observe what is termed input attribution.  In the context of autoregressive language models trained for causal language modeling (ie predicting the next token), input attribution may be thought of as the importance of each prior token in predicting that next token.

How does one take a model and find which elements in the input are most responsible for the model output?  In this case the model gives the next token in a sentence, and the input is composed of all prior tokens.  We will briefly explore two options for finding the 'most important' input elements for a given output, although this is by no means a comprehensive survey of the field of input attribution.

One method is to find the gradient of the output with respect to each input element and multiply this gradient element-wise to the input itself. This method is known as 'gradientxinput' and may be found as follows:

$$
v = | \nabla_a  O(a, \theta) | * a
$$

where the vector of inputs is $a$ and the vector of saliency values is $v$ and $* $ denotes Hadamard (element-wise) multiplication, and $\vert * \vert$ signifies the element-wise absolute value operation.  This method is intuitively similar to measuring the effect of an infinitesmal change in the input (via the gradient of the output $O(a, \theta)$ with respect to the input $a$) on the output, which a larger output change at index $i$ resulting in a larger saliency value at that index.

Another approach is to simply remove each input element sequentially and observe the change in the ouput.  If $a_c$ correponds to the input where the token at position $c$ is replaced with an informationless substitute (perhaps a $<\vert PAD \vert >$ token) then we can find a metric distance between the model's output given this masked input compared to the original $a$ as follows:

$$
v = || O(a_c, \theta) - O(a, \theta) ||_1
$$

Here we use the $L^1$ metric as a measurement of the difference between outputs, but we could just as easily have chosen any other metric. This method is appropriately termed 'occlusion' after the English word for 'make obscure'.  This is in some ways the discrete counterpoint to saliency, as here we are observing what happens to the output after a big change (replacement of an entire token).  

For small input sequences, we can form a linear combination of occlusion and gradientxinput in order to include information from both attribution methods.  For longer sequences, occlusion becomes prohibitively expensive for LLMs and therefore it is usually best to stick with saliency.

To visualize input attribution on sequences of text, we can implement an HTML-based highlighter on the decoded input tokens as follows:

```python
	def readable_interpretation(self, decoded_input, metric='combined'):
		...

		# summed_ems is a torch.tensor object of normalized input attributions per token
		positions = [i for i in range(len(summed_ems))]

		# assemble HTML file with red (high) to blue (low) attributions per token
		highlighted_text = []
		for i in range(len(positions)):
			word = decoded_input[i]
			red, green, blue = int((summed_ems[i]*255)), 110, 110
			color = '#{:02x}{:02x}{:02x}'.format(red, green, blue)
			highlighted_text.append(f'<span style="background-color: {color}">{word}</span>')

		with torch.no_grad():
			embedded_inputs = torch.clone(self.model.transformer.wte(self.input_ids))
			output = self.model(inputs_embeds=embedded_inputs)[0][:, -1, :]
			predicted_word = self.model(self.input_ids)[0][:, -1, :] # should be equal to output
			assert torch.equal(output, predicted_word)

			predicted_word = int(torch.argmax(predicted_word, dim=1))
			predicted_word = tokenizer.decode(predicted_word)

		highlighted_text = ''.join(highlighted_text)
		highlighted_text += f'</br> </br> Predicted next word: {predicted_word}'
		with open('data.html', 'wt', encoding='utf-8') as file:
			file.write(highlighted_text)
		webbrowser.open('data.html')
```

This function displays the relative attribution (importance) of each previous token by assigning the attribution value to the red highlighter color, while keeping green and blue highlighter values constant (see the snippet below).  This makes higher-attribution tokens appear red, medium-attribution orange and grey, and low-attribution tokens green-blue.

```
red, green, blue = int((summed_ems[i]*255)), 110, 110
color = '#{:02x}{:02x}{:02x}'.format(red, green, blue)
```

After loading our model of choice, we can use this module (contained in the GPTEval class, see the [source code](https://github.com/blbadger/nnetworks/blob/transformer-explorer/language_attribution.py) for more information) to generate input attribution maps as follows:

```python
if __name__ == '__main__':
	input_sequence = 'The wipers on the bus go swish swish'
	input_sequence = input_sequence.strip()
	gevaluate = GPTEval(model, input_sequence)
	input_ids = gevaluate.input_ids[0]
	decoded_input = []
	for symbol in input_ids:
		decoded_input.append(tokenizer.decode(symbol))
	arr = gevaluate.readable_interpretation(decoded_input, metric='combined')
```

which when given the prompt 'The wipers on the bus go swish swish' for the base model of GPT-2 gives

<html>
<body>
<span style="color: white">
	<span style="background-color: #156e6e">The</span><span style="background-color: #746e6e"> wip</span><span style="background-color: #006e6e">ers</span><span style="background-color: #616e6e"> on</span><span style="background-color: #626e6e"> the</span><span style="background-color: #726e6e"> bus</span><span style="background-color: #ff6e6e"> go</span><span style="background-color: #9e6e6e"> sw</span><span style="background-color: #fa6e6e">ish</span><span style="background-color: #966e6e"> sw</span><span style="background-color: #e46e6e">ish</span>
	</span>
	<br>
	<br> 
	Predicted token:  sw
</body>
</html>

Observing input attribution for larger models applied to more complicated inputs, it is clear that the gradient of a model's output with respect to its input does indeed give useful information.  This being the case, we are left with the question we had at the start of this page: why are input representations so inaccurate for trained GPT-type transformers applied to natural language, and indeed why are they less accurate than for untrained models of the same type. 

### Sequential Input Representation

Continuing our efforts to understand why trained GPT-2 models give such poor input representations compared to trained vision transformers, one can observe that there are two notable differences between language and image inputs: language is fundamentally sequential and discreet whereas images are approximately continuous and of no particular orientation.  In this section we consider the importance of the sequential aspect of language as it relates to input representation, and in the next section discreeteness is examined.

To rephrase the topic for this section, one may wonder whether or not the sequential aspect of language could contribute to difficulties in input representation.  Specifically, observe that GPT-type language models always predict one word at a time, and thus may be better at representing one word at a time too.

We can implement this sequential input representation technique as follows:

```python
def generate_sequential_input(model: torch.nn, target: torch.tensor, lr=0.5) -> torch.tensor:
	n_tokens = len(target[0])
	full_input = []
	embedding_dim = target.shape[-1]
	starting_input = torch.randn(1, 1, embedding_dim).to(device)
	combined_input = torch.clone(starting_input)
	for i in range(n_tokens):
		random_input = torch.randn(starting_input.shape).to(device)
		focused_target = target[:, i, :]
		single_input = octave(random_input, focused_target, 2000, [lr, lr/10], i) # generate single token's embedding
		combined_input = torch.cat((combined_input, single_input), dim=1)

	return combined_input[:, 1:, :]
```

For an untrained model given the input 

$$
a = \mathtt{The \; sky \; is \; blue.} 
$$

we have a top-5 input representation of

```
 Pegasus quarantine Sunshine viral Dip
 secure Help Orion Saadesh
 partic Carbon Officereller Rory
 checked regained midst ERAtile
 criticize231 SOFTWARE trunkmatically
```

and therefore we conclude that it is not the sequential characteristic of language that is the cause of poor input representation.

### Is the trained word to token embedding to blame

It might also be wondered whether the loss of input representation ability during training is due to the model learning a poorly conditioned token embedding transformation. We can test this hypothesis by replacing the trained token embedding transformation with a non-trained one, and for the direct input generation method earlier on this page this can be implemented by simply swapping the weight matrix of the word-token embedding with an untrained model's version of the same.

```python
class InputGPT(nn.Module):

	def __init__(self, model, untrained_model):
		super().__init__()
		self.model = model
		self.untrained_model = untrained_model

	def forward(self, x: torch.tensor) -> torch.tensor:
		# replaces wte transformation
		x = torch.matmul(x, self.untrained_model.transformer.wte.weight)

		for i in range(1):
			x = self.model.transformer.h[i](x)[0]
		return x
```

However, we are not met with any success in making even one transformer block yield accurate input representations.  
```
 veiledusers lesbians inquiries windshield
 dupl NeighNASAicycletre
rypted IntExp qualifying Genie
 melts funded deliber camping Mat
SHIP rejoice malepc consumers
```

Similarly poor input representations are exhibited when we use the [Roy-Moore pseudo-inverse indirect input generation method](https://blbadger.github.io/language-representations-inputs.html#sentence-representation-with-language-models) following the 

```
PsyNetMessagePsyNetMessagePsyNetMessage MarketablePsyNetMessage
 srfAttach srfAttach MarketablePsyNetMessage srfAttach
irtual unfocusedRange unfocusedRange unfocusedRange partName
 partName Marketable srfAttachawaru Marketable
ascus partName partName srfAttach unfocusedRange
```
In conclusion, we find that the training of the embedding transformation alone cannot account for the poor representation that exists in trained language models.  This does not necessarily mean that the embedding transformation does not affect input representation, but only that the process of training the embedding alone does not account for the poor input representation in GPT-2 models after training.

### Big models exhibit accurate input representations

It has been claimed that language models as they exist today are incapable of any kind of fact- or reason- based task because they are merely trained to predict the next word (or more accurately token) in some text. We can show that this claim is incorrect, however, using a fairly straightforward argument.  We can define a language model as some generalizable (to examples unseen in the training data) representation function that maps input token sequences to an output token such that the negative log-likelihood of that output token is minimized relative to the training dataset.  

From this definition alone we find that language models must also be capable of all the implicit tasks present in the training dataset.  For example, suppose the training data contained the phrase 'two plus two equals four'.  Given the input 'two plus two equals' the function must return 'four' if it has minimized the negative log-likelihood of that token. Now suppose the training data also contains the phrase 'four plus one equals five' and that now we give the unseen phrase 'two plus two plus one equals' and our function returns a token.  If our function is sufficiently generalizable, we are almost guaranteed to return 'five' because the internal representation of 'two plus two' is nearly equivalent to 'four' and the model has already learned that 'four plus one equals five'.

It has been empirically observed that models with more parameters are generally better at these implicit natural language tasks, which typically lie under the umbrella definition of 'reasoning' problems.  There are a few explanations for why this could be: firstly, larger models entail higher-dimensional space, such that gradient descent is more [biased towards generalization](https://arxiv.org/abs/2211.09639), secondly because more parameters imply greater 'memory' such that the model can learn more complicated representations of an input (ie 'three' being both a word and an implicit number) and thirdly because it is apparent that biological neural networks require a very large number of neurons to deal with language relative to other tasks, such as object recognition.

But when we have observed that although larger models (~1 billion as opposed to 100 million parameters) tend to have less repetition, they are no noticeably better at input representation than smaller models (~100M parameters). Do even larger models fail to exhibit accurate input representation?  We can investigate by observing the ability of our gradient descent procedure to autoencode an input from transformer models of a trained GPT-J model, which contains ~6B parameters.  To save memory, we can load the parameters in 8-bit format using `bitsandbytes` 

```python
load_8bit = True
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b", load_in_8bit=load_8bit, device_map='auto')
```

GPT-J, like Llama and other transformer models used today, employs Rotary Positional Encoding that is applied to each attention module (introduced by [Su and colleagues](https://arxiv.org/abs/2104.09864)).  This means that there is no `wpe` transformation to invert, but we instead have to supply the appropriate positional information as `position_ids` to each transformer block.  Using the indirect input generation method in which gradient descent is performed on the first hidden layer before the resulting tensor is inverted, the modifed version of a trained GPT-J model may be specified as follows

```python
class AbbreviatedGPT(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        position_ids = torch.tensor([i for i in range(x.shape[1])])

        for i in range(1):
            x = self.model.transformer.h[i](x, position_ids=position_ids)[0]
        return x
```

Here we investigate the representations present various layers of a model using gradient descent on the embedding of a discrete language input, a technique explained [here](https://blbadger.github.io/language-representations-inputs.html#sentence-representation-with-language-models).  For the prompt 'The sky is blue.' after $N=200$ iterations with the first transformer block of GPT-J, we generate an $a_g$ such that $\vert \vert a_g - a \vert \vert < \vert \vert a' - a \vert \vert$ which yields the top-5 token matching of

```
The skyeton resp cease
hinPresidentVER compose Instruments
uddinuko be blue_
JJzbuse Oxy rite
earlyerent iswn seasoning
```

which is certainly somewhat better than we saw for the smaller models. Note, however, that even at $N=2000$ the input representation accuracy is not sufficient to unambiguously identify the input string.

On the other hand, if we restrict the input tokens to any of 'The sky is blue or red depending on the time of day.' we come very close to recovering the input.

$$
a_g = \mathtt{The \; sky \; is \; blue \; of}
$$

Which is a notable improvement upon the smaller trained models seen previously.

This begs the question: would an even larger model be capable of even more precise input representation? We can check this using various models but one in particular is the Llama model family introduced by [Touvron and colleagues](https://arxiv.org/abs/2302.13971), which may be loaded in 8-bit parameter quantization as follows:

```python
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-13b")
model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-13b", load_in_8bit=load_8bit, device_map='auto')
```

Llama models name their components somewhat differently than GPT-type models, and apply Rotary Positional Encoding via different dimensions, so after consulting the [source code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) we subclass the model as follows:

```python
class AbbreviatedGPT(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        position_ids = torch.tensor([[i for i in range(x.shape[1])]])

        for i in range(4):
            x = self.model.model.layers[i](x, position_ids=position_ids)[0]
        return x
```

This model requires around 20GB when performing the gradient descent input representation algorithm, so the following experiments are performed using an Nvidia A100 GPU (available on Colab for a fee).

After $N=1000$ we have for the first transformer block:

$$
a_g = \mathtt{the \; sky \; the \; blue \;  }
$$

and at $N=2000$ we have the top-5 representations of

```
The sky is blue’
K Sky isn Bluerows
’sky’Blue K
The K Kativecards
rowsXrows K?"
```

and even more remarkably, at a depth of two transformer blocks for $N=2000$ the model's representation is

$$
a_g = \mathtt{The \; sky \; is \; blue.}
$$

which means that we have found a way to get accurate input representations from a trained transfomer block! Apparently the we simply have to use an extremely large model.  Thus we would expect to observe accurate input representations from even larger models, and we test this using the 30 billion parameter version of Llama (which is about the largest model that will fit in the memory of a 40GB A100 using 8-bit quantization). For the first transformer block of this trained model after a mere $N=500$ we have top-5 representations of

```
The sky is blue<s>
� Sky: Blue “
Helpsky�Blueww
smart Japan behblueATCH
cy Answer� explating
```

where `<s>` signifies a sentence break token (which is nearly identical with a period in semantic meaning).  

Compare this with the top-5 representations of the first transformer block of a 7-billion parameter version of Llama, which at $N=500$ gives

```
The sky isYS2
The ofATIONhere-
..�althOIN

,SE ${\reno
the
c blue,
```

although it should be noted that if the maximum number of gradient descent iterations $N$ increases to $N=1500$, the 7b parameter Llama yields a better

$$
a_g = \mathtt{The \; sky \; is \; blue2}
$$

With the 30 billion parameter Llama, we find that at least somewhat accurate input representations are made from deeper and deeper layers: representation after 4 transformer blocks is qualitatively similar to what is found for 1 block (above) even after 12 (!) transformer blocks, we get a recognizable input representation of

$$
a_g = \mathtt{The \; sky \; Stage \; blueInd}
$$

Note that with the 7 billion parameter version of Llama, a typical top-5 representation of the input at a depth of 12 blocks for the same $N$ is

```
havior skyichtetfiterson
burgје speakeriffsbled
prisoners prisoners standardsunct speaker
learn studying relationshipsspanburg
rium Cup containingEEali
```

meaning that we find much poorer input representations in the 7b parameter Llama as opposed to the 30b model in the deeper layers, which is to be expected with the hypothesis that larger models are capable of transmitting input information more accurately than small ones.

Returning to the 30 billion parameter model, even after 21 transformer blocks the input representation is nearly synonymous with the prompt.

$$
a_g = \mathtt{Finally \; sky \; established \; blue \; instead}
$$

This is notable because smaller language models are capable of accurate input representation before training, but only for one or at most two transformer blocks.  But with increased model size, trained models are capable of fairly accurate input representation even in relatively deep layers.

It may also be wondered whether we can recover accurate input representations by optimizing a different metric on a hidden layer output.  Earlier on this page we saw that cosine similarity used as a metric for GPT-2 models was incapable of accurate input representation, but it is worth exploring whether this is the case now that we have a larger model to work with.  As before, we perform gradient descent on the embedding $e_n$ rather than a vectorized input.

A first experiment is not promising: given the same prompt as above ('The sky is blue') after $N=500$ iterations of gradient descent on the embedding of an input such that $\phi < 0.05$ we have

$$
a_g = \mathtt{WNWNWNWN}
$$

but the slight change to 'The sky is blue.' yields

$$
a_g = \mathtt{The \; sky \; is \; blue2}
$$

Longer prompts such as 'The sky is red or blue depending on the time of day'

$$
a_g = \mathtt{The \; sky \; is  \; blue  \; or \; red \; depending \; on \; the \; time \; OF \; day}
$$

or 'This is a prompt sentence.'
 
$$
a = \mathtt{This \; is \; a \; prompt \; sentence.}
$$

We find the same tendancy for poorer input representations at deeper layers (observed above for an L1 metric distance loss) to be the case for cosine similarity loss too. For example, at block 4 of Llama 7b we have

$$
a_g = \mathtt{This \; is \; a \; prompt \; sentence \; SU}
$$

but at block 12 we have

$$
a_g = \mathtt{Thusieraución \; prompt \; Jersey \; Culture} 
$$

### Large models exhibit poor direct input representation when minimizing hidden layer L1 distance but not cosine similarity

In the last section we saw that very large models are capable of accurate indirect input representation when gradient descent performed on the input embedding, rather than directly on the input itself.  It may be wondered whether similarly accurate input representations are found from the same models if gradient descent is performed on the inputs diectly, after converting discrete token integers to a continuous vector space equivalent to those discrete tokens.

For Llama-type models that use RoPE, after converting a tokenized input into a vector-space equivalen `x` we can sub-class our chosen model as follows:

```python
class InputModel(nn.Module):

	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, x: torch.Tensor):
		# Matrix mult instead of embedding to prevent type incompatibility
		x = torch.matmul(x, self.model.model.embed_tokens.weight)
		position_ids = torch.tensor([[i for i in range(x.shape[1])]])

		for i in range(1):
			x = self.model.model.layers[i](x, position_ids=position_ids)[0]
		return x
```

Given the target input $a = \mathtt{This \; is \; a \; prompt \; sentence}$, the trained 7 billion parameter Llama model yields the following top-5 input representations at the output of the first transformer block:

```
sout spectconmicadém
Normdatei┴więひ           
Son画 hastaрук cb
ccCollectennenzen Kirchen
alusEmemed expon.<
```

We find further that the trained 30 billion parameter Llama model is similarly incapable of accurate direct input representation.  For $N$ such that $\vert \vert O(a, \theta) - O(a_g, \theta) \vert \vert < \vert \vert O(a, \theta) - O(a', \theta) \vert \vert$ we have

```
льной розта Schw университета ded
nombreuses мат FemAvailable modify
Cés projection polygonダjekt
`" помо э dll физи
dai vba грудняóp поль
```

It may be wondered whether or not a different metric would give a more accurate input representation, at least for the first transformer block of Llama 7b. 

To be more precise, we want to minimize the angle $\phi$ between the vector corresponding to the hidden layer output of some target input $a$ and a generated input $a_g$ by maximizing the cosine similarity between the output of the target input $a$, denoted $O(a, \theta)$ and the generated input $a_g$, $O(a_g, \theta)$ via gradient descent on an initially random input $a_0$.  The cosine distance may be calculated on vectorized versions of these outputs, denoted $O()^*$, as follows:

$$
\cos (\phi) = \frac{O(a, \theta)^* \cdot O(a_g, \theta)^* }{||O(a, \theta)^* ||_2 * ||O(a_g, \theta)^* ||_2}
$$

such that gradient descent on a vectorized version of the input $a_n$ as follows:

$$
a_{n+1} = a_n - \eta \nabla_{a_n} \left( 1 - \cos (\phi) \right)
$$

Maximizing the cosine similarity (ie minimizing the angle $\phi$) between generated input $a_g$ and target input $a$ yields for the $a$ given above

$$
a_g = \mathtt{This \; is \; hasta \; prompt \; sentence}
$$

where $(\cos(\phi) > 0.9$.

For 

$$
a= \mathtt{This \; is \; a \; somewhat \; longer \; prompt \; sentence.}
$$ 

we have at $\cos(\phi) = 0.76$

$$
a_g = \mathtt{This \; are \; integrated \; somewhat \; longer \; prompt \; sentence –}
$$

For another target sentence

$$
a = \mathtt{The \; sky \; is \; a \; light \; blue \; today.}
$$ 

at $\cos(\phi=0.75)$ we have

$$
a_g = \mathtt{primit \; sky \; is \; a \; light \; blue \; today \; –}
$$

It is interesting to note that cosine similarity loss does not yield accurate input representations for small inputs.  For example, given $a = \mathtt{The \; sky \; is}$ the first block representation is

$$
a_g = \mathtt{sier \; fixeseden}
$$

Nearly every one-word target input gives very inaccurate input representations, with $a_g$ as any of 'George', 'The', 'when', or 'clearly' yielding $a_g = \mathtt{que}$, and inputs with two or only a few words are often quite inaccurately representation via cosine similarity.

```
prompt = 'Boba Fett'
Nacional**** têteстори

prompt = 'Boba Fett was a bounty hunder'
фами函WNזWN поддерpersoninasacjęлище

prompt = 'Boba Fett was a legendary bounty hunter in the outer rim'
Boba Fett wasondissement legendary bounty hunter in the outer rim
```

This is a marked contrast from performing gradient descent on the input embedding, where cosine similarity yields accurate representations even for one-word inputs (ie 'George' is represented 'George').

Why would the use of cosine similarity be unable to give accurate input representations when applied to input space of very small inputs but not larger ones? It is helpful to consider here what exactly is being optimized: the cosine of $\phi$ is equal to the dot product of two vectors divided by the norms of those vectors multiplied together.

### Cosine loss optimization and model training

Why is direct input representation so inaccurate for single token inputs, even though it is accurate for multi-token inputs?  Consider that the self-attention module in the Llama transformer blocks are based on the dot product operation: the attention between any query $q$ token and a key $k$ token is given as 

$$
A(q, k, v) = \mathrm{softmax} \left( \frac{q \cdot k}{\sqrt(d)} \right) v
$$

Recalling that the dot product is equivalent to the cosine of the angle between vectors $q, k$ divided by their norms, one can say that attention compares the direction between these vectors.

But something unexpected occurs when we attempt to use cosine similarity as a metric for an untrained Llama model: for one-token inputs gradient descent is able to minimize $\cos (\phi)$ on the output of any given transformer block, but when more that one token is present in the input this is no longer possible.  

We can deconstruct the transformer block by investigating the `LlamaDecoderLayer` module from the Llama [source code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) and replicating that module in pieces as follows. 

```python
class InputGPT(nn.Module):

	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, x: torch.Tensor):
		# Matrix mult instead of embedding to prevent type incompatibility
		x = torch.matmul(x, self.model.embed_tokens.weight)
		position_ids = torch.tensor([[i for i in range(x.shape[1])]])

		for i in range(1):
			residual = x
			x = self.model.layers[i].input_layernorm(x)
			x = self.model.layers[i].self_attn(x, position_ids=position_ids)[0] + residual
			residual = x
			x = self.model.layers[i].post_attention_layernorm(x)
			x = self.model.layers[i].mlp(x) + residual
			
		return x
```

It turns out that removing the layer normalizations allows minimization of $\cos (\phi)$ on the output via gradient descent on the input, as long as the input is not too long (>5 tokens).  This is suggestive of a numerical underflow or overflow in the gradient backpropegation process, which appears to be exascerbated by the layer normalization transformations. 

Back to the question of whether untrained Llama models are capable of accurate input representation, we find that it is both more difficult to minimize $\cos(\phi)$ via gradient descent and that once minimized, the inputs are not accurate the the target $a$.  For $\cos (\phi) < 0.15$ we have

$$
a = \mathtt{The \; sky \; is \; blue} \\
a_g = \mathtt{ПерViewModeldfrac \; paths}
$$

It should be noted that untrained models appear to be worse at indirect input representation as well, as for the prompt $a = \mathtt{The \; sky \; is \; blue}$ the first transformer block from an untrained 7b Llama gives 

$$
a_g = \mathtt{leaf \; leaf  \; Connect \; leaf}
$$

at $N=1500$, whereas the trained model is much more accurate (see above).

But upon some reflection, it may not be surprising that minimizing a cosine distance between outputs of an untrained transformer block does not yield accurate input representations because the dot-product attention is followed by two fully connected layers.  If we instead observe the representation from the first ransformer block's self-attention whilst minimizing $\cos \phi$, 

```python
class InputGPT(nn.Module):

	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, x: torch.Tensor):
		position_ids = torch.tensor([[i for i in range(x.shape[1])]])

		for i in range(1):
			x = self.model.layers[i].self_attn(x, position_ids=position_ids)[0]
		return x
```
we have a permutation of the input 'The sky is blue.'

$$
a_g = \mathtt{blue \; is \; The \; sky}
$$

with a top-5 representation 

```
blue is The sky
The The sky is
sky blue is The
is sky blue blue
FLITEite период
```

and likewise for 'This is a prompt sentence' we have

$$
a_g = \mathtt{prompt. \; This. \; is \; prompt}
$$

Although with 4 self-attention blocks, more information is lost:

$$
a_g = \mathtt{sentence. \; Thisอ \; OK \; sentence}
$$

On the other hand, minimization of $L^1$ distance on the output of the first (untrained) full transformer block containing self-attention followed by an MLP gives

```
The ense The blue
sky is blue essere
pandas sky sky tör
blue land proud{$
.-- skill iseli
```

but for inputs composed of individual words (more precisely tokens) the representation is nearly perfect, for example any of the inputs 'This', 'is', 'a', 'prompt','sentence' give a representation identical to that input. As soon as the input contains multiple tokens, however, minimizing the $L^1$ distance on the output is not longer sufficient for perfectly accurate representation, for example 'the sky' is represented as 'sky sky'.

### Information between tokens

So far we have been observing the ability of the information in a transformer block's output to represent some input.  It may also be wondered whether only part of the output would suffice, as would be expected if the information from the input were to be sufficiently mixed among the transformer's tokens before reaching the output.  [Elsewhere](https://blbadger.github.io/vision-transformers.html) it was observed that vision transformers are generally much worse than attentionless MLP mixers at representing input regions not aligned to the output's token.

We first repeat the procedure used in the page linked above: we give some target input to the model and then attempt to have the model generate that input using only the information in some output layer, specifically only the first x tokens in the sequence. For Llama 7b, it is clear that this language model also does not tend to mix information in the forward direction between tokens readily: given the prompt

$$
a = \mathtt{The \; sky \; is \; blue.}
$$

For the first transformer block we have

$$
a_g = \mathtt{The \; sky \; is \; blue \; fail}
$$

but if we restrict the output to $[:, :2, :]$ we have

$$
a_g = \mathtt{The \; sky \; Sho \; advancedblock}
$$

Upon some reflection, it may be appreciated that this observation does not by itself mean that transformer-based language models do not share information between their tokens (more accurately the model's activations at each token).  This is because `llama` and other similar causal language models are trained to predict some passage's next token, such that these models should never receive information from tokens to the right of the current token.  This was not the case for vision transformers, where key, query, and value projections are performed amoung all tokens. 

Specifically causal language models perform the query-key matrix multiplications as shown in the following figure: a token's query projection is multiplied to all previous (ie left-oriented) token's value projections (and possibly its own as well). This means that information in a clm transformer's block can be influence by any token to the left of the current token, but not to the right. 

![clm explanation]({{https://blbadger.github.io}}deep-learning/clm_explanation.png)

For causal language models, therefore, we should instead check that a model's activations for tokens *after* a given token are able to specify that input token if masked. 
Effectively we can ask whether enough information passes between tokens for the language model to determine the identity of the first two tokens, given the activations present in all following tokens.  This may be done by restricting the output to $[:, 2:, :]$, which gives us

$$
a_g = \mathtt{masterlex \; is \; blue2}
$$

indicating that for this input, the information present in the last three tokens (in the output of the first transformer module) is insufficient to determine the identity of the first two tokens. The same is observed for the input 'This is a prompt sentence', which for $[:, 2:, :]$ at the first output layer gives the representation

$$
a_g = \mathtt{lex \; Sho₂ \; prompt \; sentence2}
$$

indicating that there is insufficient information transfer between later and earlier tokens to uniquely specify the masked tokens.  Similarly, taking the output of the first transformer block of all layers except the first token for the input ``Four score and seven years ago`` gives ``lex score2 seven years ago``.

It may be wondered whether a different metric might allow for more information to pass between tokens. In particular, consider the information that may pass between tokens via the multi-head attention transformation.  Recall from the last section that given vectors $q, k, v$ the attention operation is equivalent to a scaled version of the following

$$
A(q, k, v) = \mathrm {softmax} \left( q \cdot k \right) v
$$

although attention is usually calculated using matricies $Q, K, V$ such that the inner product $QK^T$ rather than the dot product is computed. Regardless, the information that passes between the query token (which for a causal language model is usually the last token) and the key token (typically one of the previous tokens).  

The dot product may be thought of as a measure of vector alignment which is a function of both distance and angle, and in that respect it is perhaps unsurprising that optimizing a pure distance norm metric would be insufficient for information transfer. Instead one could optimize angle metric, for example the cosine distance (see https://blbadger.github.io/language-discreteness.html#indirect-input-representation-via-cosine-similarity for a more thorough explanation).

Given the somewhat lengthy input prompt

$$
a = \mathtt{The \; sky \; is \; red \; or \; blue \; depending \; on \; the \; time \; of \; day.}
$$

we can test the accuracy of the input representation when the output of most but not all tokens is used to reconstruct the input. The direct input representation found cosine similarity for the trained 7 billion parameter Llama (taking the first transformer block only) for $[:, 3:, :]$ is

$$
a_g = \mathtt{XI \; Macdet \; red \; or \; blue \; depending \; on \; the \; time \; of \; day.}
$$

indicating that the information in all following tokens for this prompt is insufficient to specify any of the first three tokens. This is also the case when both cosine distance and $L^1$ distance are minimized simultaneously (via optimization of a linear combination of these measures), meaning that even if we minimize both aspects of the dot product information still does not flow sufficiently between tokens to uniquely identify them.

A little experimentation convinces one that these findings are not peculiar to the choice of input sentence but are found regardless of what input if given to the 7 billion parameter version of Llama. This observation is also not limited to the Llama model family, as Mistral 7b experiences the same incaccuracy: for [:, 1:, :] from the first transformer block we have for the input **Mario the man versus the idea**

$$
a_g = \mathtt{agenteli \; man \; versusPEGнал}
$$

and for **The sky is blue** the generated representaiton for the same output, the corresponding generated input $a_g$ is `sky<s> blue<s>`.

Being that we have seen larger models to be more and more capable at accurate input representation, it may next be wondered whether a larger model might be capable of more information transfer between tokens as well. Conceptually an arbitrarily large model would be capable of arbitrarily large information transfer between token model elements, but in particular it may be wondered if a somewhat larger version of Llama may be capable of accurate non-self token representation.

We test this by observing the ability of a model approximately twice the size of what has been used thus far: a 13 billion parameter version of Llama, again quantized to 8 bits per parameter.  We try the representation gradient descent procedure first using $L^1$ distance, for the input **This is a prompt sentence** we have ``作 is a prompt sentence`` if we use the information from all except the first token, ie $[:, 1:, :]$ from the first transformer block.

Likewise, for the 13 billion parameter version of llama when attempting input representation via cosine distance for the input 

$$
\mathtt{Mario \; the \; man \; versus \; mario \; the \; idea} 
$$

we have ``Mario the man versus mario</s> idea`` if none of the model output is masked, and

$$
\mathtt{depois \; the \; man \; versus \; mario</s> \; idea}
$$

if the first token's output is masked after the first transformer layer. Increasing the number of transformer blocks results in a less-coherent input representation and does not improve masked token identification.

For the even larger 30 billion parameter version of Llama, given the same target input as above and taking all output except that of the first token ($[:, 1:, :]$) of the first transformer block, after optimizing the $L^1$ distance on this output (using indirect input representation) the input representation $a_g$ is

$$
\mathtt{'rell \; The \; man \; versus \; mario \; The \; idea}
$$

And similarly for the input **This is a prompt sentence** we have the generated input representation $a_g$ of ``Dragon is    prompt sentence <s>`` if the output of the first token is masked for the same model. 

When we test this larger model's input representation using a cosine distance metric, we again see that there is insufficient information to uniquely identify the first token: for the target input **The sky is blue.** the model yields an $a_g$ of ``quietly sky is blue<s>`` and for **Blue is the sky.** the representation is ``quietly is The sky<s>``.

Upon testing a deeper layer (here the 8th transformer block) with the target input **This is a prompt sentence.** using an $L^1$ metric on $[:, 1:, :]$ we have

$$
a_g = \mathtt{delta \; Gray \; wer \; prompt \; sentenceInd}
$$

And for the target input **Mario the man versus Mario the idea** minimizing cosine distance on all tokens ($[:, :, :]$) we have

$$
a_g = \mathtt{Mario \; Inside \; man \; versus \; Mario \; Fund \; Mor}
$$

but when the output of the first token is masked,

$$
a_g = \mathtt{ódbaum \; man \; versus \; Mario \; processes \; idea}
$$

In the [last section](https://blbadger.github.io/language-discreteness.html#cosine-loss-optimization-and-model-output) it was observed that the cosine similarity metric could be used to find somewhat accurate representations of an input for an untrained Llama 7b, but only if the output of the attention layer rather than the output of the first transformer block (attention layer followed by two fully connected layers) was used to perform optimization upon.  It may then be wondered if we might observe a similar phenomenon here: perhaps if we optimize the cosine similarity of an attention layer, the identity of a token whose output is masked may be found. 

For untrained langauge models the representation for 7b and 30b llama is mostly the same as trained, but the untrained model exhibits one advantage: given the first transformer block's self-attention transformation alone we can find accurate non-self token input representations using $\cos \phi$ as our metric: given the prompt **Mario the idea versus Mario the man** and the outputs $[:, 1:, :]$ we have a top-5 encoding of

```
Mario man the the Mario man idea
the the man Mario versus the man
la versus versus man the Mario Mario
isi Mario Mario idea idea versusperiment
Price система idea versus man↓ouvelle
```

and for **Geralt of Rivia** we have a representation ``Ger Rivalt Rivia`` given the output of all except the first token ($[:, 1:, :]$) and ``Ger Riv of Rivia`` for all except the first two tokens, $[:, 2:, :]$. As observed before, the tokens are often permuted such that the masked token is not necessarily in the correct place. It should be noted that accurate input representations for any input tokens are not obtained once more than one attention layer is used, or once attention is followed by the MLP layers that normally comprise a transformer block.

An aside: the keen observer will not that the information from the last token is able to apparently travel to earlier tokens, which should be impossible if each attention dot-product operation only takes place in the reverse direction. This is due to the a lack of a causal mask in the attention module, which would normally be added during the causal language modeling training process.

On the other hand, some experimentation is sufficient to convince one that this is not the case for trained models: for a 7 billion parameter trained Llama, the attention layer output does not yield accurate masked token identity.  

It may be wondered why there is insufficient information passed between tokens for a trained model: which operation is required to pass information sufficiently between one layer and the next for a single token?  This is easy to answer, and removal of each transformation in the self-attention module of a trained Llama ([here](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) is the source code) shows that the removal of the residual connection across the self-attention layer is alone capable of preventing even a single input token from being correctly identified (given complete outputs) regardless of what modules follow. As there are not residual connections between token elements in transformer models, it is little surprise in that sense that there is insufficient information to correctly identify the correct input token if that token's output is masked.

Likewise, for an untrained Llama 7b we find the opposite phenomenon: removal of the residual connection across the first transformer block's self-attention layer results in little change in input representation ability: given the input 'Mario the idea versus Mario the man' we have 'Mario man the the versus man idea' without residuals and 'Mario man the the Mario man idea' with residuals intact.  Once again we find that input representation is quite inaccurate when more than one transformer layer (with or without residuals) is stacked. 

Before concluding that insufficient information passes between tokens (via attention transformations) for accurate non-self token identification, however, consider that one might have made a similar conclusion for self-token identification when using models smaller than ~1 billion parameters, but larger models (7 billion parameters and larger) do turn out to be capable of accurate self token representation (due to their increase in layer width, which parallels a communication line's bandwidth).  We may wonder if the same phenomenon would not occur for non-self tokens.

A little experimentation convinces us that an increase in model size is indeed sufficient for non-self token representation: if we initialize one transformer module of the same dimensions as the 70 billion parameter Llama-2 as follows

```python
llama_config_kwargs = {
    'hidden_size': 8192,
    'intermediate_size': 4*8192,
    'num_hidden_layers': 1,
    'num_heads': 64
}

# Initializing a LLaMA llama-70b style configuration
configuration = LlamaConfig(**llama_config_kwargs)
model = LlamaForCausalLM(configuration).to(0)
```

and using indirect input representation (minimizing an $L^2$ metric on the output) we find that the generated input for an output where the first three tokens are masked such that the output is `[:, 3:, :]` for the input **Mario the man versus Mario the idea** we have a top-2 token representation of

```
Mario the thecenwijні the
versus Mario idea man到 versusору
```

and if all tokens but the last are masked (`[:, -1, :]`) we find

```
versus census thegg canad Marioárt
```

where the non-self tokens for `Mario, versus, the` are correctly found, even if they are not in the correct token index (note that as for other randomly initialized, untrained models the exact input representation is somewhat dependent on the weight initializations).

For the trained 65 billion parameter Llama, however, the same prompt 'Mario the man versus Mario the idea' for the output of the first transformer block without the first token (`[:, 1:, :]`) representation of

$$
a_g = \mathtt{absorF \; man \; versus \; Mario \; The \; idea}
$$

which indicates that the self-token representation improves upon training, but the non-self token representation ability does not (the full `[:, :, :]` representation is `MarioThe man versus Mario The idea`). Similarly, for `[:, -1, :]` we have a_g = `absor verb ForumFOimerwho idea`, meaning only the last token is found. This is not peculiar to the chosen prompt: for example, not one of the top-5 representations for **The sky is blue.** finds the masked first token for `[:, -1, :]`,

```
ignation sky is blue?
CC Skyson Blue when
okaysky COblueriften
broke publishebrace “
bo System AskFur
```

or for different models: the trained 70 billion parameter Llama-2 gives for **Mario the idea versus the man** a top-5 representation for `[:, 1:, :]` of

```
Priceww idea versusww man
existed…aire vs…ww
港 the ideasww the Man
histor fils winningining cacheono
limitedэ mostly Pitts’ await
```

none of which correctly identifies the masked first token.  

Even if we restrict the [vocabulary](https://blbadger.github.io/language-discreteness.html#restricted-vocabulary-input-representation) such that the represented tokens may only be those that are found in the following sentence: `A sky is blue or red depending on the time of day, and further depending on the size of the vocabulary present.` for the input **The sky is blue**, we find an $a_g$ of ` size sky is blue<s>`.

It may therefore be wondered just how large a model must be in order for non-self tokens to be accurately represented. For untrained models, the chance of accurate masked token representation depends on parameter initialization but increases with model size: perhaps 1/5ths of models with a hidden dimension of 4096 (Llama 7b size) but more than 4/5ths of models with a hidden dim of 8192 (llama 70b).

To conclude, we find that there is insufficient information passing from one token to another via self-attention for trained large language models to uniquely identify inputs in which the corresponding token's hidden layer activations are masked.  The lack of information transfer between tokens is observed regardless of whether a distance, angle, or combination of distance and angle metric is used to optimize the input's representation similarity to a partially masked target. If this model is representative of others, the finding implies that transformer-based language models typically do not transfer sufficient information between tokens (via QKV matrix multiplications) for unique token identification.

This conclusion is similar to what was found for vision transformers: unlike MLP mixers (which accurately represented masked tokens), vision transformers were found to be unable to transmit sufficient information between tokens for accurate visual representation of tokens using the information in the hidden layers of other tokens.

### Mixer Information Transfer

In the last section it was observed that large untrained language models are capable of accurate non-self token representation (although the tokens may not be in the correct sequence index), but that this ability vanishes during the training process. Similar conclusions were reached [elsewhere](https://blbadger.github.io/transformer-features.html) for trained dot product attention-based transformers but curiously not for MLP mixer models that replace the self-attention operations with matrix multiplication operations. It may therefore be wondered whether an MLP mixer model might also be capable of accurate non-self token representation for language, as is the case for vision.

An implementation of a language mixer may be found on [this page](https://blbadger.github.io/smaller-lms.html), but the main idea is that one replaces the self-attention transformations with a 1-dimensional convolution (such that all feature neurons from a certain input element are multiplied by the same weight, and all these multiplied values are added together) to share weights among a token's feature neurons, but not between sequence elements. There are actually two sequential convolutions (separated by a nonlinear activation) connecting each pair of token's feature neurons, and we use a triangular mask on the convolution weights in order to allow for causal language modeling.

Note that in this section, we use a 4096-size tokenizer fitted to the TinyStories 2M dataset such that the name `Mario` now takes two tokens.

When we test the untrained mixer on both self- and non-self token representation, we find that the model size requirements for accurate input representation appear to be similar to the transformers, or perhaps slightly more : for example, given the output of all tokens (`[:, :, :]`) of the input

**Mario, the Idea, versus Mario, the Man**

One mixer block exhibits an input representation of `Mario, the Idea, versus Mario, the Man` for $d_{model}=32$, and larger. Training does not remove this accurate input representation, as for a trained model (on TinyStories 2M) we also find a perfectly accurate input representation.  

For the mixer with expanded convolutions between tokens, non-self representation is less impressive: an untrained mixer with $d_{model}=512$ with an expansion factor of 1 yields

```
it, but ario, the Idea, versus Mario, the Man
```

And the situation is not helped with training: recall for a $d_{model}=64$ mixer, we have a perfect input representation for `Mario, the Idea, versus Mario, the Man`, but we find that non-self token representation is much worse. For `[:, 1:, :]` we have (ignoring trailing tokens)

```
selessonario, the Idea, versus Mario, the Man
```

which is no better than the untrained model's representation,

```
Cario, the Idea, versus Mario, the Man
```

Similarly, a trained $d_{model}=256$ does not correctly identify the first token, and gives a representation of

```
iario, the Idea, versus Mario, the Man
```

With a non-expanded (ie 'flat') convolution between sequence elements, non-self tokens may be accurately found: for untrained mixers we have for [:, 1:, :] at $d_{model}=512$ (given the first block),

```
Mario, the Idea, versus Mario, the Man
```

This accurate self- and non-self token representation persists after training for that model, if we take the information in a slightly deeper layer (say the output `[:, 1:, :]` of the second mixer or fourth mixer block). What is more impressive is the trained flat mixer's representation given multiple masked tokens: for the output `[:, 3:, :]` of the second block we get a perfect input representation.

Reducing the hidden dimension leads to inaccurate non-self representation for the flat masked mixer, as for $d_{model}=256$ the representation is `iario, the Idea, versus Mario, the Man`. It may be wondered if a linear combination of 1-dimensional convolutions (ie in parallel rather than in series) would yield even better non-self token representation, and after some testing it does indeed: before training, if we have two parallel 1D convolutions rather than one (added together to make our linear combination) then representation is extremely acurate: given `[:, 10:, :]` of the *last* mixer block (8) for a relatively small model with $d_{model}=64$, the input **Mario, the Idea, versus Mario, the Man** is perfectly represented. 

If we look deeper into an untrained flat mixer model, on the other hand, for $d_{model}=1024$, at block 8 we find 

```
Marario, the Idea, versus Mario, the Man
```

and for $d_{model}=2048$ at block 8 we do find both accurate self and non-self token representation. This is also true if we scale the depth of the model without increasing the width, for example for $d_{model}=1024$ and $n=24$ layers we have also have a perfect self- and non-self representation for an untrained model (note that the n=8 layer version above was not as capable). What is more remarkable is that even if the first three tokens are masked, the representation is still perfect.

This flat mixer representation is more accurate than that obtained from a similarly sized transformer (even when using the same 4096-size tokenizer and training dataset): a trained $d_{model}=256, \; n=8$ transformer (llama style) model yields for `[:, 1:, :]` the input *Mario, the Idea, versus Mario, the Man*

`pictureario, the Idea, th let fterMarioiMan`

where some self- and the non-self token are incorrectly identified, although for a trained $d_{model}=512$ we have

`s. They whistch whistsat panstayou're snowpatophch whistsat Man`

### Noise on a Discreet Channel

To recap, we have found that accurate input representations of language but not images are not formed in trained transformer models unless they contain a very large number of parameters, particularly in trained models.  In the next section, we will consider what this means for a language model's ability to give useful outputs.

But first it remains to be seen why training would result in worse input representations, why larger models would be capable of much more accurate representation, and above all why accurate input representation appears to be so much more difficult for language than for images.

In developing the theory of communication over a noisy channel, Shannon and others found a mathematical explanation for a phenomenon that initially seemed most curious: the fact that the amount of information reaching the receiver often decreases imperceptably at first and then suddenly plummets as the amount of information increases (given constant noise).  

Given that deep learning models often behave as if they were noisy communication channels, it may be wonderd if the same observation would be made for these models.  Indeed it is found that language model input representation experiences a severe drop as the channel width decreases. For the 7 billion parameter trained Llama with 4096 feature neurons per layer, reducing this number to 2202 yields no incorrect input tokens for the input 'Mario the man versus Mario the idea' but decreasing this by even one neuron (ie taking `[:, :, :2201]` as the output used to perform gradient descent-based input optimization) leads to *no* tokens being correctly represented. The same is true when we consider only the attention layers in modules (no MLPs), where the number of necesary neurons now is 2083.  These observations are not due to the identity of the specific tokens being chosen, but can be thought of as a true bandwidth phenomenon: in the case of the attention-only module taking `[:, :, 1000:2083]` yields no token accurately found but taking `[:, :, 1000:3083]` gives every input token accurately.

### Implications of Representation Accuracy

Modern large language models are usually trained by first predicting the next word in a sentence, followed by what is termed 'aligmnent' which serves to make the language model return outputs are appropriate for some given at hand, which could be helpfully answering questions or perhaps providing background information.  This alignment is usually achieved via supervised fine-tuning, deep reinforcement learning, or a combination of these two approaches. 

It should be appreciated that in some sense most the fundamental problem of language processing, the task of producing gramatically correct language, does not require a very large model at all or even a very sophisticated architecture.  More recently it has become desirable for language models to be able to perform tasks that are intrinsic to language (factual recall, reasoning etc.) but these were not the tasks that were the original goal of current architectures, particularly the transformer which was originally identified as a language model that resisted saturation (defined here as a lack of improvement in perplexity after training on additional tokens). This is not a unique phenomenon to transformers, however, as the same is observed for large convolutional and even fully connected architectures.

Therefore one can wonder whether the transformer is as effective an architecture for meta-language tasks as it is for language tasks. 

In spite of these innovations, it has been observed that models smaller than around 10 billion parameters (using commonly applied scaling measures for MLP and key, query and value projection parameters in transformers) are generally insufficient for anything but the simplest of language tasks.

Does it matter for the purposes of language generation that even otherwise effective models are incapable of differentiating between nonsensical gibberish and meaningful sentences?  At first glance it may seem as though it may not matter: if a language model were to be given these nonsenical phrases then it may confuse them with actual text, but what are the chances that the exact nonsensical phrase would appear as a prompt to a language model in a real application?  

There is, however, reason to wonder whether it is not important that language models form such poor representations of their inputs.  Language models as they currently exist suffer from a significant and currently difficult-to-manage problem sometimes referred to as 'hallucinations', in which the model will return syntactically and semantically correct text that is woefully incorrect in the implicit language task at hand.  Furthermore, at present there appears to be no method that is capable of preventing this hallucination barring directly training against specific examples (either using supervised or reinforcement methods).  

This is fundamentally a problem of representation: if a language model were capable of representing all necessary implicit and explicit language tasks and inputs to a sufficient degree of accuracy, the model would be capable of discerning text that fails to address the implicit tasks from text that does not fail to do so.  As we have already seen that language models cannot represent their inputs uniquely, it may be little wonder why they are sometimes incapable of representing implicit input features as well.
















