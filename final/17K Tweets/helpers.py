import pandas as pd
import random
import re
from nltk import WordPunctTokenizer
from snowballstemmer import TurkishStemmer

def readLineByLine(data):
    tweets = []
    for row in range(data.shape[0]):    
        tweet = []

        for col in range(data.shape[1]):        
            tweet.append(data.iat[row, col])

        tweets.append(tweet)

    return tweets

def createCsv():
    train_tweets = readLineByLine(pd.read_excel("train_tweets.xlsx"))
    test_tweets = readLineByLine(pd.read_excel("test_tweets.xlsx"))

    all_tweets = train_tweets + test_tweets
    random.shuffle(all_tweets)

    data = pd.DataFrame(all_tweets, columns=['Sentence', 'Sentiment'])
    data.to_csv('17k-tweets.csv', index=False)

def handleEmojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    
    return tweet
        
def cleanForTweet(tweet):
    # Replaces URLs with the word URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', '', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', '', tweet)
    # Replaces #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    
    return tweet

def cleanNormalText(sentence):
    # Remove all the special characters
    sentence = re.sub(r'\W', ' ', sentence)
    # Remove all digit characters
    sentence= re.sub(r'\d', '', sentence)
    # remove all single characters
    sentence= re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)
    # Remove single characters from the start
    sentence = re.sub(r'\^[a-zA-Z]\s+', ' ', sentence) 
    # Substituting multiple spaces with single space
    sentence = re.sub(r'\s+', ' ', sentence, flags=re.I)

    return sentence

def get_external_stopwords():
    file = open("stop_words.txt", "r", encoding='utf8')
    stop_words = [word.strip() for word in file] 
    file.close() 
    
    return stop_words
   
def filter_stop_words(text, stop_words):
    wpt = WordPunctTokenizer()
    tokenized_words = wpt.tokenize(text)
    processed_words = [word for word in tokenized_words if not word in stop_words]
    text = ' '.join([str(word) for word in processed_words])
    
    return text
    
def stemming_words(text):    
    wpt = WordPunctTokenizer()
    words = wpt.tokenize(text)
    
    turkishStemmer = TurkishStemmer()
    
    stemmed_words = []
    for word in words:
        stemmed_words.append(turkishStemmer.stemWord(word))
    text = ' '.join([str(word) for word in stemmed_words])  
    
#     print (stemmed_words)
    
    return text 

def find_max_length(features):
    length = 0    
    for sentence in features:
        if len(sentence) > length:
            length = len(sentence)
    return length
    
def cleanFeatures(features, stop_words):
    processed_features = []

    for index in range(0, len(features)): 
        
        # Convert to lower case
        processed_feature = str(features[index]).lower()
        
        # Replace emojis with either EMO_POS or EMO_NEG
        processed_feature = handleEmojis(processed_feature)
        
        # Cleaning sentence for tweets
        processed_feature = cleanForTweet(processed_feature)
        
        # Cleaning sentence for normal texts
        processed_feature = cleanNormalText(processed_feature)        
        
        # Cleaning stop words         
        processed_feature = filter_stop_words(processed_feature, stop_words)        
        
        # Stemming words     
        # processed_feature = stemming_words(processed_feature)
        
        processed_features.append(processed_feature)
    
    return processed_features  