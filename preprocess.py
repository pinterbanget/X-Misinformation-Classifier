#%%
import pandas as pd
import numpy as np
import nltk
import os
import re
import pickle
import fasttext


from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
DetectorFactory.seed = 0 # for reproducibility

from lingua import LanguageDetectorBuilder, Language
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from datetime import datetime

#%%
# Fasttext model
fasttext_model = fasttext.load_model('lid.176.bin')

# Lingua model
lingua_detector = LanguageDetectorBuilder.from_all_languages().build()

# Text preprocessing and language detection functions
def preprocess_text(text, language='english'):
    """
    Preprocess text with the following steps:
    1. Lowercase conversion
    2. Remove URLs, special characters, and numbers
    3. Tokenization
    4. Remove stopwords
    5. Lemmatization
    Returns: Preprocessed tokens
    """
    # 1. Convert to lowercase
    text = text.casefold()
    
    # 2. Remove URLs, special characters, and numbers
    text = re.sub(r'https?://\S+|www\.\S+', '[LINK]', text)  # URLs
    text = re.sub(r'(?<!\[LINK\])[^a-zA-Z\s]', '', text)  # Non-alphabetic characters
    text = re.sub(r'\d+', '', text)  # Numbers
    
    # 3. Tokenize text
    tokens = word_tokenize(text)
    
    # 4. Remove stopwords
    stop_words = set(stopwords.words(language))
    tokens = [word for word in tokens if word not in stop_words]
    
    # 5. Lemmatization
    # lemmatizer = WordNetLemmatizer()
    # tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens

def is_english_lingua(text):
    return lingua_detector.detect_language_of(text) == Language.ENGLISH

def is_english_fasttext(text):
    """
    Returns True if the tweet is classified as English, False otherwise.
    """
    # FastText's predict method returns a tuple:
    # (labels, probabilities)
    labels, probabilities = fasttext_model.predict(text)
    
    # The label has the form '__label__en' for English
    language_code = labels[0].replace('__label__', '')
    
    return language_code == 'en'

def is_english_langdetect(text):
    ''' 
    Detect if a text is written in English
    Returns: True if the text is in English, False otherwise
    TODO: Instead of just using langdetect, use an ensemble of language detection methods
    '''
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False
    
def is_english_ensemble(text):
    text = text.replace('\n', '')
    return is_english_langdetect(text) & is_english_lingua(text) & is_english_fasttext(text)
#%%
if __name__ == '__main__':
    print(os.getcwd())
    post_list = [pd.read_csv("data/" + file, sep="\t") for file in os.listdir("data")]
    df = pd.concat(post_list, ignore_index=True)
    df = df.dropna(subset=["tweetText", "summary"])
    print(f'Total length of original dataset: {len(df)}')
    
    df['is_english'] = df['tweetText'].apply(is_english_ensemble)

    english_df = df[df['is_english'] == True]
    print(f'Total length of cleaned dataset: {len(english_df)}')

    english_df.to_csv('english_tweets.tsv', index=False, sep='\t'
                      )
    X = english_df['tweetText'].tolist()
    y = english_df['classification'].to_list()
    X_community_note = english_df['summary'].tolist()
    
    preprocessed_data = {}
    
    preprocessed_data["X_preprocessed"] = [preprocess_text(i) for i in X]
    preprocessed_data["y_processed"] = [1 if i == 'MISINFORMED_OR_POTENTIALLY_MISLEADING' else 0 for i in y]
    preprocessed_data["X_community_note"] = [preprocess_text(i) for i in X_community_note]

    print("Preprocessed:", preprocessed_data["X_preprocessed"][:2])
    print("Processed label: ", preprocessed_data["y_processed"][:2])
    print("Preprocessed Community Note: ", preprocessed_data["X_community_note"][:2])
    print(f'Total misleading posts: {sum(preprocessed_data['y_processed'])}')
    
    path = f'clean_data/complete_plus_community_note_{datetime.now().strftime("%Y%m%d")}.pkl'
    
    with open(path, "wb") as f:
        pickle.dump(preprocessed_data, f)

    print(f"saved to pickle file in: {path}")


    