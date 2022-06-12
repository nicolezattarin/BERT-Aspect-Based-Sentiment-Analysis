import pandas as pd
import numpy as np

def tag_to_word(sentence, predictions):
    """
    predictions: list of tags
    sentence: list of words
    """
    terms = []
    for i, word in enumerate(sentence):
        if predictions[i] == 1:
            terms.append(word)
    return terms

def tag_to_word_df(df, column_name, tags):
    """
    predictions: list of tags
    sentence: list of words
    """
    terms_list = []
    for i in range(len(df)):
        sentence = df.iloc[i]['Tokens']
        sentence = sentence.replace("'", "").strip("][").split(', ')
        terms = tag_to_word(sentence, tags[i])
        terms_list.append(terms)
    df[column_name] = terms_list
    return df