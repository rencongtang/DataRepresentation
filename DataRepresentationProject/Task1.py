# Part 1 data cleaning
# In this part, my target is to make process of my data
# To achieve my target, I need to write 3 methods below:
# 1. make tokenization of the data: first separate the data into sentences and each sentence is a list of words
# 2. remove stop words in the each sentences
# 3. use stem on each word in each sentence
from csv_helper import load_csv
from nltk.tokenize import TweetTokenizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords


tweets = load_csv("Tweets_2016London.csv")


def Tokenization(tweets):
    tknzer = TweetTokenizer()
    tokens = list([])
    for tweet in tweets:
        token = tknzer.tokenize(tweet)
        tokens.append(token)
    return tokens


def StopWordsRemoval(sentence):
    stopwords_list = stopwords.words('english')
    content = list([])
    for word in sentence:
        if word.lower() not in stopwords_list:
            content.append(word)

    return content


def Stemming(sentence):
    st = LancasterStemmer()
    content = list([])
    for word in sentence:
        new_word = st.stem(word)
        content.append(new_word)
    return content

tweets_tokenization = Tokenization(tweets)
tweet_clean = list([])
for tweet in tweets_tokenization:
    tweet_svr = StopWordsRemoval(tweet)
    tweet_s = Stemming(tweet_svr)
    tweet_clean.append(tweet_s)


print(tweet_clean)

# Part 2
