import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences

# Ensure stopwords and lemmatizer are available
# Ensure stopwords and lemmatizer are available
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Replace mentions and hashtags
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Tokenize and remove stopwords
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def filter_tweets(tweets, min_length=5):
    """
    Filters out tweets that are too short after preprocessing.
    """
    filtered_tweets = []
    for tweet in tweets:
        cleaned_tweet = preprocess(tweet)
        if len(cleaned_tweet.split()) >= min_length:
            filtered_tweets.append(cleaned_tweet)
    return filtered_tweets


# Example tweets
tweets = [
    "@Thomas1774Paine @JoeBiden #DOJ@TheJusticeDept #INSTRUCTING ?! #SUPREME#COURT ?! #DO#NOT#BLOCK #NEW#EVICTION #MORATORIUM ?! ~ ~ #AMERICANS #DEPEND !! ON #EARNINGS FROM THESE #INVESTMENTS !! ~ ~ @JoeBiden &amp; @TheJusticeDept #ADMITTED #UNCONSTITUTIONAL !!",
    "Congress: PLEASE Extend Eviction Moratorium for tens of millions of struggling Americans. With everything going on right now..You can not put people out on the streets. Its straight up wrong. DO SOMETHING. @TheDemocrats @POTUS @SpeakerPelosi @SenSchumer @SenSanders @JeffBezos",
    "@POTUS Biden Blunders - 6 Month Update Inflation, Delta mismanagement, COVID for kids, Abandoning Americans in Afghanistan, Arming the Taliban, S. Border crisis, Breaking job growth, Abuse of power (Many Exec Orders, $3.5T through Reconciliation, Eviction Moratorium)...what did I miss?"
]

# Apply preprocessing and filter short tweets
cleaned_tweets = filter_tweets(tweets)

# Print cleaned and filtered tweets
for i, tweet in enumerate(cleaned_tweets, 1):
    print(f"Cleaned and Filtered Tweet {i}: {tweet}")

# Tokenize and pad sequences for further processing
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(cleaned_tweets)
sequences = tokenizer.texts_to_sequences(cleaned_tweets)
padded_sequences = pad_sequences(sequences, maxlen=200)

# Print tokenized and padded sequences
print("Padded Sequences:")
print(padded_sequences)