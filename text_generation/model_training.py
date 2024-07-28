import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
import keras

def prepare_sequences(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    sequences = pad_sequences(sequences, maxlen=50)

    input_sequences = []
    for sequence in sequences:
        for i in range(1, len(sequence)):
            n_gram_sequence = sequence[:i+1]
            input_sequences.append(n_gram_sequence)

    max_sequence_len = max(len(x) for x in input_sequences)
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    return input_sequences, max_sequence_len, tokenizer

def train_model(input_sequences, max_sequence_len, tokenizer):
    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = keras.utils.to_categorical(label, num_classes=len(tokenizer.word_index) + 1)

    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_sequence_len - 1),
        LSTM(150),
        Dense(len(tokenizer.word_index) + 1, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(predictors, label, epochs=10, verbose=1)
    return model
