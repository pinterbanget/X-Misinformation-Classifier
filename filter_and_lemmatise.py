#%%
import pandas as pd
import pickle

from nltk.stem import WordNetLemmatizer

lemmatiser = WordNetLemmatizer()

def tweet_longer_than_length(tweet, length=100):
    return len(tweet) >= length

def lemmatise_tweet(tweet, lemmatiser):
    return ' '.join([lemmatiser.lemmatize(word) for word in tweet.split()])

#%%
if __name__ == '__main__':
    df = pd.read_csv('english_tweets.tsv', sep='\t')
    
    # Filter out tweets that are too short
    df['sufficient_length'] = df['tweetText'].apply(tweet_longer_than_length)
    df = df[df['sufficient_length'] == True]

    # Lemmatise tweets
    df['tweetText'] = df['tweetText'].apply(lemmatise_tweet, lemmatiser=lemmatiser)
    
    preprocessed_data = {}
    
    preprocessed_data["X"] = df['tweetText'].tolist()
    preprocessed_data["y"] = [1 if i == 'MISINFORMED_OR_POTENTIALLY_MISLEADING' else 0 for i in df['classification'].tolist()]
    preprocessed_data["X_notes"] = df['summary'].tolist()
    
    path = f'filtered_data.pkl'
    
    with open(path, "wb") as f:
        pickle.dump(preprocessed_data, f)

    print(f"Saved to pickle file in: {path}")


    