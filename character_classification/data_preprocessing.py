import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(filepath):
    return pd.read_json(filepath, lines=True)

def preprocess_text(data):
    data['text'] = data['text'].apply(lambda x: x.lower())
    return data

def vectorize_text(data):
    vectorizer = CountVectorizer(stop_words='english')
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['speaker'], test_size=0.2, random_state=42)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, y_train, y_test

def prepare_data_nn(data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data['text'])
    X = tokenizer.texts_to_sequences(data['text'])
    X = pad_sequences(X, maxlen=100)
    y = pd.get_dummies(data['speaker']).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    return X_train, X_test, y_train, y_test, tokenizer
