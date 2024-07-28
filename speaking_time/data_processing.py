import pandas as pd

def load_data(filepath):
    return pd.read_json(filepath, lines=True)

def filter_data(df):
    main_characters = ['Monica Geller', 'Joey Tribbiani', 'Chandler Bing', 'Phoebe Buffay', 'Ross Geller', 'Rachel Green']
    return df[df['speaker'].isin(main_characters)].copy()

def extract_season(df):
    df.loc[:, 'season'] = df['id'].apply(lambda x: x.split('_')[0])
    return df

def create_contingency_table(df):
    return df.groupby(['season', 'speaker']).size().unstack(fill_value=0)
