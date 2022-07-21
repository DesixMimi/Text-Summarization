# Text Summarization

[cnn_dailymail - tensorflow](https://www.tensorflow.org/datasets/catalog/cnn_dailymail)
## CNN Daily Dataset
CNN-dailymail dataset is a collection of the articles mostly news, interviews that have been published on the two popular websites CNN.com and dailymail.com.
It contains more than 200 million words.
Like the common styles on newspapers and journals, each article contains some highlighted sections that together form the summary of the whole article.
This dataset that originally has been gathered in the work of Hermann et al., 2015, has become a standard source for training and evaluating text summarizer models.

[Taming Recurrent Neural Networks for Better Summarization](http://www.abigailsee.com/2017/04/16/taming-rnns-for-better-summarization.html)

## What is text summarization?
Text summarization is the problem of reducing the number of sentences and words of a document without changing its meaning.

There are two model methods used:
* **Extractive** - select the most important sentences within a text (without necessarily understanding the meaning).
* **Abstractive** - uses advanced NLP (i.e. word embeddings) to understand the semantics of the text and generate a meaningful summary.


# Steps for and notes for building Text Summarization Model
1. **Reading the data**

2. **Data Preprocessing** - A mere flaw in a dataset can ruin hours and days
of training procedure. For example in a sequence-to-sequence problem where the
data consists of a pair of source and target, pairs are sorted and indexed in two same size lists. Shifting, adding or removing a single record in each list will damage the balance of all the pairs so that each source instance will bind to a wrong target instance.
* Articles which are too long (outliers) should be truncated.
* Some redundant character and symbols, hyperlinks, HTML tags and non-latin alphabets should be removed.

3. **Check the distribution of the Data**
4. **Tokenize the Data** - A tokenizer builds the vocabulary and converts a word sequence to an integer sequence.
5. **Model building** 
> Important parameters:

  >**Return Sequences = True**: When the return sequences parameter is set to True, LSTM produces the hidden state and cell state for every timestep.

  > **Return State = True:** When return state = True, LSTM produces the hidden state and cell state of the last timestep only
Initial State: This is used to initialize the internal states of the LSTM for the first timestep.

  >**Stacked LSTM**: Stacked LSTM has multiple layers of LSTM stacked on top of each other. This leads to a better representation of the sequence. I encourage you to experiment with the multiple layers of the LSTM stacked on top of each other (it’s a great way to learn this).

6. **Training and Early Stopping** - Stopping to train the model when we see that after a certain epoch the loss stops to decrease.

7. **Inference** - Set up the inference for the encoder and decoder. Here the Encoder and the Decoder will work together, to produce a summary. The Decoder will be stacked above the Encoder, and the output of the decoder will be again fed into the decoder to produce the next word.
>How does the inference process work?
  Here are the steps to decode the test sequence:
  1. Encode the entire input sequence and initialize the decoder with internal states of the encoder.
  2. Pass <**start**> token as an input to the decoder
  3. Run the decoder for one timestep with the internal states
  4. The output will be the probability for the next word. The word with the maximum probability will be selected
  5. Pass the sampled word as an input to the decoder in the next timestep and update the internal states with the current time step
  6. Repeat steps 3 – 5 until we generate <**end**> token or hit the maximum length of the target sequence

8. **Testing**

# Summery of Models 

## Long Short-Term Memory (LSTM)

### What is LSTM?
Long Short-Term Memory (LSTM) networks are a modified version of recurrent neural networks (RNN), they were invented to solve the vanishing gradient problem of RNN by saving the past data in memory.


>*Another RNN's issue was with long range memory ( the memory of the first inputs gradually fades away) when dealing with textual data and having deep NN, for example, when dealing with the task of predicting the next word of a long sentence* 

### The main idea
Each single LSTM cell governs what to remember, what to forget and how to update the memory using gates. By doing so, the LSTM network solves the problem of exploding or vanishing gradients, as well as other problems.

### LSTM gates
In an LSTM network, there are three gates:
* **Input gate** — discover which value from input should be used to modify the memory. *Sigmoid* function decides which values to let through 0,1. and *tanh* function gives weightage to the values which are passed deciding their level of importance ranging from-1 to 1.
* **Forget gate** — discover what details to be discarded from the block. It is decided by the sigmoid function. it looks at the previous state(ht-1) and the content input(Xt) and outputs a number between 0(omit this)and 1(keep this)for each number in the cell state Ct−1.
* **Output gate** — the input and the memory of the block is used to decide the output. *Sigmoid* function decides which values to let through 0,1. and *tanh* function gives weightage to the values which are passed deciding their level of importance ranging from-1 to 1 and multiplied with output of Sigmoid.

Credit to [Michel Kana](https://towardsdatascience.com/5-secrets-about-lstm-and-gru-everyone-else-know-97446d89e35b)

[Tensorflow Keras LSTM source code line-by-line explained](https://medium.com/softmax/tensorflow-keras-lstm-source-code-line-by-line-explained-125a6dae0622)

### LSTM's downsides
* takes longer to train
* very long gradient paths
* require more memory to train
* LSTMs are easy to overfit
* Dropout is much harder to implement in LSTMs
* LSTMs are sensitive to different random weight initializations

[Credic](https://datascience.stackexchange.com/questions/27392/so-whats-the-catch-with-lstm)


## Sequence to Sequence (Seq2Seq)
### what is Seq2Seq?
Seq2Seq model is a neural net that transforms a given sequence of elements, such as the sequence of words in a sentence, into another sequence.

> Sentences, for example, are sequence-dependent since the order of the words is crucial for understanding the sentence. LSTM are a natural choice for this type of data.

Seq2Seq models consist of an **Encoder** and a **Decoder**. The Encoder takes the input sequence and maps it into a higher dimensional space (n-dimensional vector). That abstract vector is fed into the Decoder which turns it into an output sequence. The output sequence can be in another language, symbols, a copy of the input, etc.

The bottleneck of the model is where the real magic happens. The information that is fed into the model is squeezed to a vector of desired length and this squeezed vector in the bottleneck stores important information that the decoder uses to predict the output. Often this vector is called the **latent vector**.

### Embedding vs Latent Space
"Latent space" and "embedding" both refer to an (often lower-dimensional) representation of high-dimensional data:

* Latent space refers specifically to the space from which the low-dimensional representation is drawn.
* Embedding refers to the way the low-dimensional data is mapped to ("embedded in") the original higher dimensional space.
[credit](https://ai.stackexchange.com/questions/11285/what-is-the-difference-between-latent-and-embedding-spaces)

### What is Attention and how is related?
The attention-mechanism looks at an input sequence and decides at each step which other parts of the sequence are important.

Instead of only writing down the translation of the sentence in the imaginary language, the Encoder also writes down keywords that are important to the semantics of the sentence, and gives them to the Decoder in addition to the regular translation. Those new keywords make the translation much easier for the Decoder because it knows what parts of the sentence are important and which key terms give the sentence context.

In other words, for each input that the LSTM (Encoder) reads, the attention-mechanism takes into account several other inputs at the same time and decides which ones are important by attributing different weights to those inputs. The Decoder will then take as input the encoded sentence and the weights provided by the attention-mechanism. 

## Transformers
### What is a Trnsformer?
Like LSTM, Transformer is an architecture for transforming one sequence into another one with the help of two parts (Encoder and Decoder), but it differs from the previously described/existing sequence-to-sequence models because it does not imply any RNN (GRU, LSTM, etc.) only with attention-mechanisms.

### The architecture of Transformers
Both Encoder and Decoder are composed of modules that can be stacked on top of each other multiple times, they consist mainly of Multi-Head Attention and Feed Forward layers.

One slight but important part of the model is the positional encoding of the different words. Since we have no recurrent networks that can remember how sequences are fed into a model, **we need to somehow give every word/part in our sequence a relative position** since a sequence depends on the order of its elements.

>*full desciprion can be found in the medium article in the credit below*

### Notes for training the Transformer
* We should always shift the decoders input because we do not want our model to learn how to copy our decoder input during training, since the target word/character for position i would be the word/character i in the decoder input.
* We fill the first position of the decoder input with a start-of-sentence token and append an end-of-sentence token to the decoder input sequence to mark the end.
* The Transformer applies a mask to the input in the first multi-head attention module to avoid seeing potential ‘future’ sequence elements. 
* The target sequence we want for our loss calculations is simply the decoder input without shifting.

Credit to [Maxime](https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04)

## Link to google colab with more details: [Text Summarization- TF](https://colab.research.google.com/drive/1NKQKt-2hNOnOtqPB8MWetqBnzNgf8Spp?usp=sharing)

