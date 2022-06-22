from matplotlib.transforms import Bbox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def tag_to_word(sentence, predictions):
    """
    predictions: list of tags
    sentence: list of words
    """
    terms = []
    for i, word in enumerate(sentence):
        w = None
        if predictions[i] == 1:
            w = word 
            for j in range(i+1, len(sentence)):
                if predictions[i] == 2:
                    w += ' ' + sentence[i+1]
                else: 
                    terms.append(w)
                    i = j
                    break

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


def word_cloud (data):
    from wordcloud import WordCloud
    wordcloud = WordCloud(collocations=False, 
                          background_color="cornflowerblue",
                          colormap="tab10",
                          max_words=50).generate(data)

    return wordcloud


def target_predicted_wordcloud(targets, predicted, file_name):
    
    sns.set_theme(style='white', font_scale=1.5)
    fig, ax = plt.subplots(1, 2, figsize=(22, 6))
    ax[0].imshow(word_cloud(targets))
    ax[0].axis("off")
    ax[0].set_title("Target")
    ax[1].imshow(word_cloud(predicted))
    ax[1].axis("off")
    ax[1].set_title("Predicted")
    fig.savefig(file_name, dpi=300, bbox_inches='tight')
