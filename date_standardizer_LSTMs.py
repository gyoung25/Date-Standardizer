'''This code is based on an assignment from the DeepLearning.AI's Coursera course titled Sequence Models. I have expanded on the assignment by writing two models with different architectures and writing a helper function that facilitates model evaluation.

Skills demonstrated: LSTMs, attention, RNNs, NLP, tensorflow, data processing, model evaluation, general coding

Overview: 

> Long-Short Term Memory (LSTM) networks are powerful tools for natural language processing (NLP) tasks. They expand on basic recurrent neural network (RNN) models by encoding a "memory" that addresses the vanishing gradient problem that persists in classic RNNs. Attention models expand on traditional RNNs by passing more information between encoder and decoder layers, thereby allowing the model to learn important features connecting parts of text that appear more than a few words apart. This code contains three LSTM models: one without attention and a more traditional architecture, one with attention, and one with a naive approximation to attention that simply uses fully connected layers to pass information between the encoder and decoder layers.

Models:

1. LSTM without attention. Encoder layer is a bidirectional LSTM. Output from the last timestep is fed into a (unidirectional) LSTM decoder layer.
2. LSTM with attention. Encoder layer is a bidirectional LSTM. Output <i> from each time step </i> **and** the hidden state from the previous time step of the decoder layer is fed into an attention layer, which passes the LSTM output and hidden state through a dense layer, then through a softmax layer that is used to calculate the context. The context is then fed into the decoder layer, which is a (unidirectional) LSTM.
3. LSTM with approximated attention. Similar to the LSTM model with attention, but the attention layer does not receive the hidden state from the decoder layer and passes the output of the dense layer directly to the decoder layer.

Purpose: 
- Demonstrate the effectiveness of LSTMs with attention performing NLP tasks.

Datasets: 

- A randomly generated dataset containing a prescribed number of dates in a variety of human-readable formats

Target:

- The given date in machine-readable format. (Assumptions: If no day is specified in the human-readable date, assign the day as the first of the month. If no month is specified, assign January.)
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from tensorflow.keras.layers import RepeatVector, Dense, Activation, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import time

from faker import Faker
import random
from tqdm import tqdm
#from babel.dates import format_date
from date_std_tools import *

#Generate data
m = 20000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)

#Define the max length of human-readable dates (Tx) and machine readable dates (Ty)
#In the context of the model, Tx is the max character length of an input string and Ty is the length of the output
Tx = 20
Ty = 10

#X (Y): lists of length Tx representing each human-readable (machine-readable) date. Dimension (m,Tx) ((m,Ty))
#Xoh (Yoh): one-hot representation of X (Y). Dimension (m,Tx,len(human_vocab)) ((m,Ty,len(machine_vocab)))
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("Xoh.shape:", Xoh.shape)
print("Yoh.shape:", Yoh.shape)

#Define LSTM layer sizes
n_a = 32 # number of units for the bi-directional LSTM's hidden state a
n_s = 64 # number of units for the uni-directional LSTM's hidden state s
num_epochs = 3 #number of epochs to train each model

#--------------------- Model with attention ---------------------#

# Define shared layers as global variables for model_with_attention
repeat = RepeatVector(Tx)
concat = Concatenate(axis=-1)
dense1 = Dense(10, activation = "relu")
dense2 = Dense(1, activation = "relu")
activate = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dot = Dot(axes = 1)

post_attention_LSTM = LSTM(n_s, return_state = True)
softmax_layer_attn = Dense(len(machine_vocab), activation=softmax)

#Define model_with_attention
model_attn = model_with_attention(Tx, Ty, n_a, n_s,
                                  len(human_vocab), len(machine_vocab),
                                  repeat, concat, dense1, dense2, activate, dot,
                                  post_attention_LSTM, softmax_layer_attn)
opt = Adam(learning_rate=0.005, weight_decay=0.005) # Adam(...) 
model_attn.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))

start_attn = time.time()
model_attn.fit([Xoh, s0, c0], outputs, epochs=num_epochs, batch_size=64)
end_attn = time.time()

#model_attn.summary()

#--------------------- Model with approximate attention ---------------------#

#Define shared LSTM layer for model_approx_attention
LSTM_approx = LSTM(units=n_s, return_state = True)
softmax_layer_approx = Dense(len(machine_vocab), activation=softmax)

#Define model with approximate attention
model_approx_attn = model_approx_attention(Tx, Ty, n_a, n_s,
                                           len(human_vocab), len(machine_vocab),
                                           LSTM_approx, softmax_layer_approx)

opt = Adam(learning_rate=0.005, weight_decay=0.005) # Adam(...) 
model_approx_attn.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))

start_approx = time.time()
model_approx_attn.fit([Xoh, s0, c0], outputs, epochs=num_epochs, batch_size=64)
end_approx = time.time()

#model_approx_attn.summary()

#--------------------- Model without attention ----------------------------#

#Define shared LSTM layer for model_just_LSTM
LSTM_just = LSTM(units=n_s, return_state = True)
softmax_layer_just = Dense(len(machine_vocab), activation=softmax)

#Define model without attention
model_just_LSTM = model_just_LSTM(Tx, Ty, n_a, n_s,
                                  len(human_vocab), len(machine_vocab),
                                  LSTM_just, softmax_layer_just) 

opt = Adam(learning_rate=0.005, weight_decay=0.005) # Adam(...) 
model_just_LSTM.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))

start_just = time.time()
model_just_LSTM.fit([Xoh, s0, c0], outputs, epochs=num_epochs, batch_size=64)
end_just = time.time()

#model_just_LSTM.summary()

#----------------- Evaluate model(s) performance using dates.txt ----------------#

#load list of human-readable dates from dates.txt
doc = []
with open('dates.txt') as file:
    for line in file:
        doc.append(line)

doc = [x.lower().strip().replace(',','').replace('.','') for x in doc]

acc_attn, wrong_attn, _ = model_performance_test(model_attn, doc, human_vocab, inv_machine_vocab, Tx, n_s)
print('LSTM with attention:')
print('Time to train ' + str(num_epochs) + ' epochs : ' + f"{end_attn-start_attn:.2f}" + ' seconds.')
print('Accuracy: ' + str(acc_attn))
print('Number of dates incorrectly predicted: ' + str(len(wrong_attn)))
print('Incorrectly predicted dates in the form (Input date, predicted date, actual date):\n' + str(wrong_attn))

acc_approx, wrong_approx, _ = model_performance_test(model_approx_attn, doc, human_vocab, inv_machine_vocab, Tx, n_s)
print('LSTM with approximated attention:')
print('Time to train ' + str(num_epochs) + ' epochs: ' + f"{end_approx-start_approx:.2f}" + ' seconds.')
print('Accuracy: ' + str(acc_approx))
print('Number of dates incorrectly predicted: ' + str(len(wrong_approx)))
print('Incorrectly predicted dates in the form (Input date, predicted date, actual date):\n' + str(wrong_approx))

acc_just, wrong_just, _ = model_performance_test(model_just_LSTM, doc, human_vocab, inv_machine_vocab, Tx, n_s)
print('LSTM model, no attention:')
print('Time to train ' + str(num_epochs) + 'epochs : ' + f"{end_just-start_just:.2f}" + ' seconds.')
print('Accuracy: ' + str(acc_just))
print('Number of dates incorrectly predicted: ' + str(len(wrong_just)))
print('Sample incorrectly predicted dates in the form (Input date, predicted date, actual date):\n' + str(wrong_just[:10]))

#----------- Evaluate model performance on dates with typos -------------#

doc_misspell = ['decembre 3, 2000', 'merch 1986', 'juen, 17, 1999', '29 fbreuary 2024', 'novmebr 35 20100']

m_pred = len(doc_misspell)
s00 = np.zeros((m_pred, n_s))
c00 = np.zeros((m_pred, n_s))
_, Xoh_pred = prepare_input(doc_misspell, human_vocab, Tx)
    
pred_attn = model_attn.predict([Xoh_pred, s00, c00])
pred_approx = model_approx_attn.predict([Xoh_pred, s00, c00])
pred_just = model_just_LSTM.predict([Xoh_pred, s00, c00])
    
#convert output of the model into a list of human-readable dates
output_attn = prepare_output(pred_attn, inv_machine_vocab)
output_approx = prepare_output(pred_approx, inv_machine_vocab)
output_just = prepare_output(pred_just, inv_machine_vocab)


#Create a list of 4-tuples of (human_readable, model_attn_predicted, 
# model_approx_attn_predicted, model_just_LSTMs_predicted)
input_output = list(zip(doc_misspell, output_attn, output_approx, output_just))

print('LSTM with attention:')
print('Predicted dates in the form (Input date, model_attn predicted date, model_approx_attn predicted, model_just_LSTMs predicted):\n')
for x in input_output:
    print(str(x)+'\n')