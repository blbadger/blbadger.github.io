
## Language Modeling and Discrete Encodings

This page is a continuation of Parts [I](https://blbadger.github.io/language-representations.html) and [II]((https://blbadger.github.io/language-representations-inputs.html)). 

### Introduction

In [Part II](https://blbadger.github.io/language-representations-inputs.html) it was observed that hidden layer representations of the model input are typically hopelessly inaccurate, which is somewhat surprising being that vision transformers and convolutional models are capable of quite accurate input representation even in deeper layers.  This page begins by further testing representation accuracy before exploring the theory behind poor input representation for language models and concludes with a study on how representation can be more accurate and how how this accuracy affects language tasks.

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

After loading our model of choice, we can use the GPTEval class to generate input attribution maps as follows:

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

Continuing our efforts to understand why trained language transformers give such poor input representations compared to trained vision transformers, one can observe that there are two notable differences between language and image inputs: language is fundamentally sequential and discreet whereas images are approximately continuous and of no particular orientation.  In this section we consider the importance of the sequential aspect of language as it relates to input representation, and in the next section discreeteness is examined.

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

For an untrained model given the input 'The sky is blue.' we have a top-5 input representation of

```
 Pegasus quarantine Sunshine viral Dip
 secure Help Orion Saadesh
 partic Carbon Officereller Rory
 checked regained midst ERAtile
 criticize231 SOFTWARE trunkmatically

```

### Is the trained word to token embedding to blame

For the direct input generation method introduced above, this can be implemented by simply swapping the weight matrix of the word-token embedding with an untrained model's version of the same.

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

Similarly poor input representations are exhibited when we use the [Roy-Moore pseudo-inverse indirect input generation method](https://blbadger.github.io/language-representations-inputs.html#sentence-representation-with-language-models)  

```
PsyNetMessagePsyNetMessagePsyNetMessage MarketablePsyNetMessage
 srfAttach srfAttach MarketablePsyNetMessage srfAttach
irtual unfocusedRange unfocusedRange unfocusedRange partName
 partName Marketable srfAttachawaru Marketable
ascus partName partName srfAttach unfocusedRange

```

### Noise on a Discreet Channel

### Implicit Language Tasks

It has been claimed that language models as they exist today are incapable of any kind of fact- or reason- based task because they are merely trained to predict the next word (or more accurately token) in some text. We can show that this claim is incorrect, however, using a fairly straightforward argument.  We can define a language model as some generalizable (to examples unseen in the training data) representation function that maps input token sequences to an output token such that the negative log-likelihood of that output token is minimized relative to the training dataset.  

From this definition alone we find that language models must also be capable of all the implicit tasks present in the training dataset.  For example, suppose the training data contained the phrase 'two plus two equals four'.  Given the input 'two plus two equals' the function must return 'four' if it has minimized the negative log-likelihood of that token. Now suppose the training data also contains the phrase 'four plus one equals five' and that now we give the unseen phrase 'two plus two plus one equals' and our function returns a token.  If our function is sufficiently generalizable, we are almost guaranteed to return 'five' because the internal representation of 'two plus two' is nearly equivalent to 'four' and the model has already learned that 'four plus one equals five'.

It has been empirically observed that models with more parameters are generally better at these implicit natural language tasks, which typically lie under the umbrella definition of 'reasoning' problems.  There are a few explanations for why this could be: firstly, larger models entail higher-dimensional space, such that gradient descent is more [biased towards generalization](https://arxiv.org/abs/2211.09639), secondly because more parameters imply greater 'memory' such that the model can learn more complicated representations of an input (ie 'three' being both a word and an implicit number) and thirdly because it is apparent that biological neural networks require a very large number of neurons to deal with language relative to other tasks, such as object recognition.

But when we have observed that although larger models (>1b parameters) tend to have less repetition, they are no noticeably better at input representation than smaller models (~100M parameters). Do even larger models fail to exhibit accurate input representation?  We can investigate by observing the ability of our gradient descent procedure to autoencode an input from transformer models of a trained GPT-J model, which contains ~6B parameters.  To save memory, we can load the parameters in 8-bit format using `bitsandbytes` 

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

For the prompt 'The sky is blue.' after $N=200$ iterations with the first transformer block of GPT-J, we generate an $a_g$ such that $\vert \vert a_g - a \vert \vert < \vert \vert a' - a \vert \vert$ which yields the top-5 token matching of

```
The skyeton resp cease
hinPresidentVER compose Instruments
uddinuko be blue_
JJzbuse Oxy rite
earlyerent iswn seasoning
```

which is certainly somewhat better than we saw for the smaller models. Note, however, that even at $N=2000$ we do not see 

On the other hand, if we restrict the input tokens to any of 'The sky is blue or red depending on the time of day.' we come very close to recovering the input.

$$
a_g = \mathtt{The \; sky \; is \; blue \; of}
$$

Which is a notable improvement upon the trained models seen previously.

This begs the question: would an even larger model be capable of even more precise input representation? We can check this using various models but one in particular is the Llama family, which may be loaded in 8-bit parameter quantization as follows:

```python
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-13b")
model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-13b", load_in_8bit=load_8bit, device_map='auto')
```
Llama models name their components somewhat differently than GPT-type models, and apply Rotary Positional Encoding via different dimensions, so after consulting the [source code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) we subclass the model as follwos:
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

This model requires around 19 GB when performing the gradient descent-based input representation algorithm
After $N=1000$ we have for the first transformer block:

$$
\mathtt{the \; sky \; the \; blue \;  }
$$

and at $N=2000$ we have the top-5 representations of

```
The sky is blue’
K Sky isn Bluerows
’sky’Blue K
The K Kativecards
rowsXrows K?"
```

and even more remarkably, after two transformer blocks for $N=2000$ we have

$$
a_g = \mathtt{The \; sky \; is \; blue.}
$$

which means that we have found a way to get accurate input representations from a trained transfomer block! We simply have to use an extremely large model.

If simply making a transformer-based language model larger and training it on more text is not able to lead to models becoming better able to represent their inputs, then what is? Language models today often follow general training in which the sole metric is predicting the next word in a sentence with what is termed 'aligmnent' which serves to make the language model return outputs that are aligned in some way to the task at hand (question answer, mathematics, etc.).  This alignment is usually achieved via supervised fine-tuning, deep reinforcement learning, or a combination of these two approaches.  

Does it matter for the purposes of language generation that even otherwise effective models are incapable of differentiating between nonsensical gibberish and meaningful sentences?  At first glance it may seem as though it may not matter: if a language model were to be given these nonsenical phrases then it may confuse them with actual text, but what are the chances that the exact nonsensical phrase would appear as a prompt to a language model in a real application?  

There is, however, reason to wonder whether it is not important that language models form such poor representations of their inputs.  Language models as they currently exist suffer from a significant and currently difficult-to-manage problem sometimes referred to as 'hallucinations', in which the model will return syntactically and semantically correct text that is woefully incorrect in the implicit language task at hand.  Furthermore, at present there appears to be no method that is capable of preventing this hallucination barring directly training against specific examples (either using supervised or reinforcement methods).  

This is fundamentally a problem of representation: if a language model were capable of representing all necessary implicit and explicit language tasks and inputs to a sufficient degree of accuracy, the model would be capable of discerning text that fails to address the implicit tasks from text that does not fail to do so.  As we have already seen that language models cannot represent their inputs uniquely, it may be little wonder why they are sometimes incapable of representing implicit input features as well.






















