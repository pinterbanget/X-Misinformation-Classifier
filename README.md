# Early Flagging of Misleading Posts on X (formerly Twitter)

## Overview

This project aims to develop a system that can identify and flag misleading posts on X (formerly Twitter) using publicly-available <a href="https://x.com/i/communitynotes/download-data">Community Notes</a> data before they gain significant traction. The system will use a combination of machine learning and deep learning techniques to detect and flag misleading content.

## Novelty

Our novelty lies in two aspects:

1) We are the first to use Community Notes data to flag misleading posts on X.
2) We propose a novel approach to incorporate Community Notes data by building a two-tower neural network model inspired by recommendation systems.

## Methodology

### Data Collection

- <b>Community Notes Data</b>: We use the Community Notes data directly from <a href="https://x.com/i/communitynotes/download-data">X</a> to retrieve every single Community Note ever created. We also utilise the Notes rating data to filter out low-quality notes and notes related to media content.
- <b>X Post Data</b>: We use a combination of X API, web scraping, and manual addition to retrieve the posts that appear in the Community Notes data. During data collection, an ensemble of language models are used to filter out the posts that are not in English.

### Data Preprocessing

- <b>Length Filtering</b>: We filter out posts that are too short (less than 100 characters).
- <b>Lemmatising</b>: We lemmatise the text using the <a href="https://www.nltk.org/">nltk</a> library.
- <b>Lowercasing</b>: We convert all text to lowercase.
- <b>Punctuation Removal</b>: We remove punctuation and special characters using regular expressions.
- <b>Stopword Removal</b>: We remove stopwords using the <a href="https://www.nltk.org/">nltk</a> library.
- <b>Tokenisation</b>: We tokenise the text using the <a href="https://www.nltk.org/">nltk</a> library.

### Model Design

For models that only use post data:
- <b>Embedding</b>: Multiple embedding models are explored: Bag-of-Words, TF-IDF, Word2Vec, and built-in tokenisers from BART and BERTweet models.
- <b>Machine Learning Models</b>: We explore two machine learning models: Support Vector Machine (baseline, with TF-IDF embeddings) and CatBoost.
- <b>Deep Learning Models</b>: We use BART and BERTweet embedding layers, in conjunction with a bidirectional LSTM layer, to create a deep learning model.

For models that use both post and note data:
- <b>Single-tower Neural Network</b>: We build a single-tower neural network model. The input is the concatenation of the post and note embeddings. The output is a final score. Knowledge distillation is also explored.
- <b>Two-tower Neural Network</b>: We build a two-tower neural network model inspired by recommendation systems. The two towers are the Post tower and the Note tower. The Post tower is used to encode the post text, while the Note tower is used to encode the note text. The two towers' outputs are then handled with a fusion layer to produce a final score.

We focus on Class 1 (Misleading) Recall as the most important metric, as we want to flag as many misleading posts as possible.

## Results

When only using post data, the best performing model is the CatBoost model with Bag-of-Words embeddings. 

When using both post and note data, the best performing model:
- when not including note data on the test set: the single-tower BART-LSTM neural network model,
- when including note data: the two-tower BART-LSTM neural network model.

## Reproducibility

`english_tweets.tsv` is the dataset used for the experiments. The dataset is based on X's Community Notes data until March 11, 2025; note that some posts are not retrievable due to either the poster deleting the post, the post being deleted by X, and the poster privating their account.

To reproduce the results:

1) Run `filter_and_lemmatise.py` to create `filtered_data.pkl`, ready to load on all the other code.
2) For the machine learning models, run `machine_learning_models.ipynb`. Ensure that all libraries are installed.
3) For the deep learning models:
    - Files with `two_tower` in the name are for the two-tower neural network models.
    - `knowdis` indicated the knowledge distillation models.