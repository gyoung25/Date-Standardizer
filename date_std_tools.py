import numpy as np
from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from tensorflow.keras.layers import RepeatVector, Dense, Activation, Lambda

fake = Faker()
Faker.seed(12345)
random.seed(12345)

# Define format of the data we would like to generate
FORMATS = ['short',
           'medium',
           'long',
           'full',
           'full',
           'full',
           'd MMM YYY', 
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'MMMM dd YYY',
           'MMMM dd, YYY',
           'MM/dd/YY',
           'MM/d/YY',
           'MM/dd/YYYY',
           'MM/dd/YYYY',
           'MM/dd/YYYY'
           'M/dd/YYYY'
           'M/dd/YYYY',
           'M/d/YYYY'
           'M/d/YYYY'
           'MM/d/YYYY',
           'MM/d/YYYY',
           'MM-d-YY',
           'MM-d-YY',
           'MM-dd-YY',
           'MM-dd-YY',
           'M-d-YY',
           'M-d-YY',
           'M-dd-YY',
           'M-dd-YY',
           'MMM yyyy',
           'MMM yyyy',
           'MM/yyyy',
           'MM/yyyy',
           'MM/yyyy',
           'M/yyyy',
           'M/yyyy',
           'M/yyyy',
           'yyyy',
           'yyyy',
           'yyyy']

# change this if you want it to work with another language
LOCALES = ['en_US']

def load_date():
    """
        Loads some fake dates 
        :returns: tuple containing human readable string, machine readable string, and date object
    """
    dt = fake.date_object()
    dt_format = random.choice(FORMATS)
    if 'd' not in set([*dt_format]) and not any(dt_format is x for x in ['full','long','medium','short']):
        dt = dt.replace(day=1) 
    if 'M' not in set([*dt_format]) and not any(dt_format is x for x in ['full','long','medium','short']):
        dt = dt.replace(month=1) 
    
    try:
        human_readable = format_date(dt, format=dt_format,  locale='en_US') # locale=random.choice(LOCALES))
        human_readable = human_readable.lower()
        human_readable = human_readable.replace(',','')
        machine_readable = dt.isoformat()
        
    except AttributeError as e:
        return None, None, None

    return human_readable, machine_readable, dt

def load_dataset(m):
    """
        Loads a dataset with m examples and vocabularies
        Arguments
            m: the number of examples to generate
        Returns
            dataset: list of ordered pairs containing randomly generated dates.
                     Each ordered pair is in the form (human-readable, machine-readable)
            human: dictionary mapping each character found among all human-readable dates to a unique integer
            machine: dictionary mapping each character found among all machine-readable dates to a unique integer
            inv_machine: dictionary inverting the mapping defined in the dictionary machine
    """
    
    human_vocab = set()
    machine_vocab = set()
    dataset = []
    Tx = 30
    

    for i in tqdm(range(m)):
        h, m, _ = load_date()
        if h is not None:
            dataset.append((h, m))
            human_vocab.update(tuple(h))
            machine_vocab.update(tuple(m))
    
    human = dict(zip(sorted(human_vocab) + ['<unk>', '<pad>'], 
                     list(range(len(human_vocab) + 2))))
    inv_machine = dict(enumerate(sorted(machine_vocab)))
    machine = {v:k for k,v in inv_machine.items()}
 
    return dataset, human, machine, inv_machine

def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty):
    
    """
    Returns integer sequence and one-hot representations of text features in the dataset.
    Note: The same preprocessing could be done using tensorflow's built-in TextVectorization and CategoryEncoding layers
    
    Arguments
        dataset: list of tuples containing (human_readable_date, machine_readable_date)
        human_vocab: dictionary of characters used in the human-readable dates
        machine_vocab: dictionary of characters used in machine-readable dates
        Tx, Ty: the max length of strings of human-readable and machine-readable dates, respectively
    Returns
        X: (len(dataset), Tx) dimensional numpy array, each row of which is an array of integers
           representing the position of the characters in the human-readable date vocabulary
        Y: (len(dataset), Ty) dimensional numpy array, each row of which is an array of integers
           representing the position of the characters in the machine-readable date vocabulary
        Xoh, Yoh: one-hot representations of X and Y, of dimensions (len(dataset), Tx, len(human_vocab)) and 
           (len(dataset), Ty, len(machine_vocab)), respectively
    """
    X, Y = zip(*dataset)
    
    X = np.array([string_to_int(i, Tx, human_vocab) for i in X])
    Y = np.array([string_to_int(t, Ty, machine_vocab) for t in Y])
    
    Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), X)))
    Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(machine_vocab)), Y)))



    return X, Y, Xoh, Yoh

def prepare_input(X, human_vocab, Tx):
    """
    Very similar to preprocess_data, only designed specifically for the prediction dataset
    Arguments
        X: list of human-readable dates
        human_vocab: dictionary of characters used in the human-readable dates
        Tx: the max length of strings of human-readable dates
    Returns
        X_pred: (len(dataset), Tx) dimensional numpy array, each row of which is an array of integers
           representing the position of the characters in the human-readable date vocabulary
        Xoh_pred: one-hot representations of X_pred of dimension (len(X), Tx, len(human_vocab))
    """
    X_pred = np.array([string_to_int(i, Tx, human_vocab) for i in X])
    Xoh_pred = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), X_pred)))
    
    return X_pred, Xoh_pred

def prepare_output(pred, inv_vocab):
    """
    Converts the output of the model into a list of human-readable dates
    Arguments
        pred: predictions from the LSMT attention model (list of softmax outputs of length Ty for each training example)
        inv_vocab: dictionary mapping machine readable indexes to machine readable characters
    Returns
        output: list of strings in human-readable dates converted from the LSTM prediction
    """
    pred = np.argmax(pred, axis = -1)
    output = [int_to_string(x, inv_vocab) for x in pred.swapaxes(0,1)]
    output = [''.join(x) for x in output]
    
    return output

def string_to_int(string, length, vocab):
    """
    Converts all strings in the vocabulary into a list of integers representing the positions of the
    input string's characters in the "vocab"
    
    Arguments:
    string -- input string, e.g. 'Wed 10 Jul 2007'
    length -- the number of time steps you'd like, determines if the output will be padded or cut
    vocab -- vocabulary, dictionary used to index every character of your "string"
    
    Returns:
    rep -- list of integers (or '<unk>') (size = length) representing the position of the string's character in the vocabulary
    """
    
    #make lower to standardize
    string = string.lower()
    string = string.replace(',','')
    
    if len(string) > length:
        string = string[:length]
        
    rep = list(map(lambda x: vocab.get(x, '<unk>'), string))
    
    if len(string) < length:
        rep += [vocab['<pad>']] * (length - len(string))
    
    #print (rep)
    return rep


def int_to_string(ints, inv_vocab):
    """
    Output a machine readable list of characters based on a list of indexes in the machine's vocabulary
    
    Arguments:
    ints -- list of integers representing indexes in the machine's vocabulary
    inv_vocab -- dictionary mapping machine readable indexes to machine readable characters 
    
    Returns:
    l -- list of characters corresponding to the indexes of ints thanks to the inv_vocab mapping
    """
    
    l = [inv_vocab[i] for i in ints]
    return l


def softmax(x, axis=1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')
        
def attention_step(a, s_prev, repeat, concat, dense1, dense2, activate, dot):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.
    
    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    
    Returns:
    context -- context vector, input of the next (post-attention) LSTM cell
    """
    # Use repeat layer to repeat s_prev to be of shape (m, Tx, n_s) so it can be concatenated with all hidden states a
    s_prev = repeat(s_prev)
    # Use concatenate layer to concatenate a and s_prev on the last axis
    conc = concat([a, s_prev])
    # Use dense1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e.
    e = dense1(conc)
    # Use dense2 to propagate e through a small fully-connected neural network to compute the energies variable energies.
    energies = dense2(e)
    # Pass energies through activation layer to compute the attention weights alphas
    alphas = activate(energies)
    # dot together with alphas and a to compute the context vector to be given to the next (post-attention) LSTM-cell
    context = dot([alphas, a])
    ### END CODE HERE ###
    
    return context

def model_with_attention(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size, repeat, concat, dense1, dense2, activate, dot, post_attention_LSTM, softmax_layer):
    """
    Returns an LSTM model with attention. Architecture is briefly summarized as 
    [Bidirectional LSTM layer] -> ([Attention layer] -> [(Unidirectional) LSTM layer])(Ty times) -> [Softmax layer].
    
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"
    repeat -- RepeatVector layer (used in attention step)
    concat -- Concatenate layer (used in attention step)
    dense1 -- Dense layer (used in attention step)
    dense2 -- Dense layer (used in attention step)
    activate -- softmax layer (used in attention step)
    post_attention_LSTM -- unidrectional LSTM layer following the attention layer
    softmax_layer -- softmax layer to transform the output of post_attention_LSTM into a prediction

    Returns:
    model -- Keras model instance
    """
    from tensorflow.keras.layers import Bidirectional, Concatenate, Input, LSTM
    from tensorflow.keras.models import Model
    #from tensorflow.keras.layers import RepeatVector, Dense, Activation, Lambda

    # Define the inputs of your model with a shape (Tx, human_vocab_size)
    # Define s0 (initial hidden state) and c0 (initial cell state)
    # for the decoder LSTM with shape (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    # initial hidden state
    s0 = Input(shape=(n_s,), name='s0')
    # initial cell state
    c0 = Input(shape=(n_s,), name='c0')
    # hidden state
    s = s0
    # cell state
    c = c0
    
    # Initialize empty list of outputs
    outputs = []

    
    # Define pre-attention Bi-LSTM
    a = Bidirectional(LSTM(units=n_a, return_sequences=True))(X)

    # Iterate for Ty steps
    for t in range(Ty):
    
        # Call attention function to context vector
        context = attention_step(a, s, repeat, concat, dense1, dense2, activate, dot)
        
        #Pass context, hidden state s, and cell state c into post-activation LSTM cell
        _, s, c = post_attention_LSTM(inputs=context, initial_state=[s, c])
        
        #Apply dense layer to the hidden state output of the post-attention LSTM
        out = softmax_layer(s)
        
        # Append the output of the dense layer to the outputs list
        outputs.append(out)
    
    # Create model
    model = Model(inputs=[X,s0,c0], outputs=outputs)
    
    
    return model

def model_approx_attention(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size, LSTM_layer, softmax_layer):
    """
    Returns an LSTM model with a naive approximation to an attention layer. Architecture is briefly summarized as 
    
    [Bidirectional LSTM layer] -> ([Dense layer] -> [(Unidirectional) LSTM layer])(Ty times) -> [Softmax layer].
    
    The interior dense layer plays a role similar to an attention layer in that it takes in the output sequence of the bidirectional LSTM and transforms it into a tensor that is fed into the LSTM_layer. Note that, unlike the attention model, this interior dense layer does not take as input the hidden state of the previous LSTM_layer step.
    
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the second LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"
    LSTM_layer -- LSTM layer that transforms the output of a dense layer, used to make prediction
    softmax_layer -- softmax layer to transform the output of post_attention_LSTM into a prediction
    
    Returns:
    model -- Keras model instance
    """
    
    from tensorflow.keras.layers import Bidirectional, Concatenate, Input, LSTM
    from tensorflow.keras.models import Model
    
    # Define the inputs of your model with a shape (Tx, human_vocab_size)
    # Define s0 (initial hidden state) and c0 (initial cell state)
    # for the decoder LSTM with shape (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    # initial hidden state
    s0 = Input(shape=(n_s,), name='s0')
    # initial cell state
    c0 = Input(shape=(n_s,), name='c0')
    # hidden state
    s = s0
    # cell state
    c = c0
    
    # Initialize empty list of outputs
    outputs = []

            
    # Define pre-attention Bi-LSTM
    a = Bidirectional(LSTM(units=n_a, return_sequences=True))(X)
    
    # Iterate for Ty steps
    for t in range(Ty):
        
        context = Dense(16, activation='tanh')(a)
        
        #Pass context, hidden state s, and cell state c into post-activation LSTM cell
        _, s, c = LSTM_layer(inputs=context, initial_state=[s,c])
        
        #Apply dense layer to the hidden state output of the post-attention LSTM
        out = softmax_layer(s)
        
        # Append the output of the dense layer to the outputs list
        outputs.append(out)
    
    # Create model
    model = Model(inputs=[X,s0,c0], outputs=outputs)
    
    return model
        
    
def model_just_LSTM(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size, LSTM_layer, softmax_layer):
    """
    Returns an LSTM model without activation. Architecture is briefly summarized as 
    
    [Bidirectional LSTM layer] -> [(Unidirectional) LSTM layer])(Ty times) -> [Softmax layer].
    
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the second LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"
    LSTM_layer -- LSTM layer that transforms the output of a dense layer, used to make prediction
    softmax_layer -- softmax layer to transform the output of post_attention_LSTM into a prediction

    Returns:
    model -- Keras model instance
    """
    
    from tensorflow.keras.layers import Bidirectional, Concatenate, Input, LSTM
    from tensorflow.keras.models import Model
    import tensorflow as tf
    
    # Define the inputs of your model with a shape (Tx, human_vocab_size)
    # Define s0 (initial hidden state) and c0 (initial cell state)
    # for the decoder LSTM with shape (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    # initial hidden state
    s0 = Input(shape=(n_s,), name='s0')
    # initial cell state
    c0 = Input(shape=(n_s,), name='c0')
    # hidden state
    s = s0
    # cell state
    c = c0
    
    # Initialize empty list of outputs
    outputs = []

            
    # Define pre-attention Bi-LSTM
    a = Bidirectional(LSTM(units=n_a))(X)
    a = tf.expand_dims(a, axis=1)

    # Iterate for Ty steps
    for t in range(Ty):
        
        #Pass context, hidden state s, and cell state c into post-activation LSTM cell
        _, s, c = LSTM_layer(inputs=a, initial_state=[s,c])
        
        #Apply dense layer to the hidden state output of the post-attention LSTM
        out = softmax_layer(s)
        
        # Append the output of the dense layer to the outputs list
        outputs.append(out)
    
    # Create model
    model = Model(inputs=[X, s0, c0], outputs=outputs)
    
    return model

def model_performance_test(model, doc, human_vocab, inv_machine_vocab, Tx, n_s, ):
    '''
    Arguments
        model: any of the LSTM models defined in this code
        doc: list of strings of human-readable dates
        human_vocab: dictionary of characters used in the human-readable dates
        inv_machine_vocab: dictionary inverting the mapping defined in the dictionary machine
        Tx: the max length of strings of human-readable dates
        n_s: hidden state size of the second LSTM
    Returns
        acc: accuracy score, (# correct)/(total number)
        wrong: A list of 3-tuples of the form (human-readable date, model-predicted date, pandas-predicted date)
               where the model-predicted and pandas-predicted dates are not the same
        input_output: Full list of 3-tuples of the form (human-readable date, model-predicted date, pandas-predicted date)
    '''
    import pandas as pd
    
    m_pred = len(doc)
    s00 = np.zeros((m_pred, n_s))
    c00 = np.zeros((m_pred, n_s))
    _, Xoh_pred = prepare_input(doc, human_vocab, Tx)
    
    prediction = model.predict([Xoh_pred, s00, c00])
    
    #convert output of the model into a list of human-readable dates
    output = prepare_output(prediction, inv_machine_vocab)
    
    #Use pandas.to_datetime to convert the human-readable dates from doc into machine-readable form
    dt_output = list(pd.to_datetime(doc,yearfirst=True).strftime('%Y-%m-%d'))
    
    #Create a list of ordered triplets of (human_readable, model_predicted_machine_readable, pandas.to_datetime)
    input_output = list(zip(doc,output,dt_output))
    
    #Find all dates that the model converted incorrectly and store in a list
    wrong = [(x,y,z) for x,y,z in input_output if y != z]
    
    acc = 1-len(wrong)/m_pred
    return acc, wrong, input_output