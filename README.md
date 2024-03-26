# Date-Finder

This repository contains two key pieces of code: 
  1. dateFinder.py searches through a txt file, extracts all dates using RegEx, returns them as a pandas Series,         and optionally writes a new txt file containing the dates separated by line breaks (\n)
  2. date_standardizer_LSTMs.py trains three LSTM-based models to convert human-readable dates in any format into        machine-readable format. Details of the model architectures can be found in the .py file, but briefly, one is       a standard LSTM model, one is an LSTM model with attention, and one is an LSTM with a simple approximation to       attention.

<u>dateFinder.py</u>

This code is based on an assignment from the University of Michigan's Coursera course titled Appled Text Mining in Python.

This code is intended to demonstrate a nontrivial use of RegEx for text data extraction.

The code looks through a dataset of 500 typed medical notes and extracts dates from each.

The formatting of the dates in the notes varies quite a bit. For example, dates could be written 08/01/2023; 8/1/23; August 1, 2013; Aug 1, 2013; Aug. 1, 2013; 1 Aug 2013; Aug 2013 (no day); 2023 (no day or month), etc.

Assumptions: dates missing a day will be assigned the first day of the month, 01. Dates missing a day and month will be assigned January first of the given year.

<u>date_standardizer_LSTMs.py</u>

This code is based on an assignment from the DeepLearning.AI's Coursera course titled Sequence Models. I have expanded on the assignment by writing two models with different architectures and writing a helper function that facilitates model evaluation.

**Skills demonstrated:** LSTMs, attention, RNNs, NLP, tensorflow, data processing, model evaluation, general coding

Overview: 

> Long-Short Term Memory (LSTM) networks are powerful tools for natural language processing (NLP) tasks. They expand on basic recurrent neural network (RNN) models by encoding a "memory" that addresses the vanishing gradient problem that persists in classic RNNs. Attention models expand on traditional RNNs by passing more information between encoder and decoder layers, thereby allowing the model to learn important features connecting parts of text that appear more than a few words apart. This code contains three LSTM models: one without attention and a more traditional architecture, one with attention, and one with a naive approximation to attention that simply uses fully connected layers to pass information between the encoder and decoder layers.

Models:

> 1. LSTM without attention. Encoder layer is a bidirectional LSTM. Output from the last timestep is fed into a (unidirectional) LSTM decoder layer.
2. LSTM with attention. Encoder layer is a bidirectional LSTM. Output <i> from each time step </i> **and** the hidden state from the previous time step of the decoder layer is fed into an attention layer, which passes the LSTM output and hidden state through a dense layer, then through a softmax layer that is used to calculate the context. The context is then fed into the decoder layer, which is a (unidirectional) LSTM.
3. LSTM with approximated attention. Similar to the LSTM model with attention, but the attention layer does not receive the hidden state from the decoder layer and passes the output of the dense layer directly to the decoder layer.

Purpose: 
> Demonstrate the effectiveness of LSTMs with attention performing NLP tasks.

Datasets: 

> A randomly generated dataset containing a prescribed number of dates in a variety of human-readable formats

Target:

> The given date in machine-readable format. (Assumptions: If no day is specified in the human-readable date, assign the day as the first of the month. If no month is specified, assign January.)
