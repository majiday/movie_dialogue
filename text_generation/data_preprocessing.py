import pandas as pd

def load_data(filepath):
    return pd.read_json(filepath, lines=True)

def preprocess_text(data):
    data['text'] = data['text'].apply(lambda x: x.lower())  # Lowercasing
    return data
