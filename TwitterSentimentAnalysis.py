
#imports
from nltk import pos_tag, classify
from nltk.tokenize import word_tokenize
from nltk.corpus import twitter_samples, stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.classify.naivebayes import NaiveBayesClassifier


# https://www.nltk.org/howto/corpus.html
#positive = twitter_samples.strings('positive_tweets.json')
#negative = twitter_samples.strings('negative_tweets.json')


"""put stop words into set and in english"""
stop_words = set(stopwords.words('english'))

from typing import List


def clean(tokens: List[str]) -> List[str]:
    '''clean tokens by removing stopwords, lowercasing, and lemmatizing'''
    cleaned_tokens = []
    for token in tokens:
        if token not in stop_words:
            cleaned_tokens.append(token.lower())

    lemmantized = []
    for word, tag in pos_tag(cleaned_tokens):
        """returns orginal word and tag pos = part of speech"""
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmantized.append(WordNetLemmatizer().lemmatize(word, pos))
        """lemmnatize = chopped off prefix and postfix"""

    return lemmantized

"""Getting the positive and negative tweets as tokenized"""
positive_token = twitter_samples.tokenized('positive_tweets.json')
negative_token = twitter_samples.tokenized('negative_tweets.json')

"""Cleaning tweets"""
positive_cleaned = [clean(t) for t in positive_token]
negative_cleaned = [clean(t) for t in negative_token]


"""Converting cleaned tweets to datasets"""
positive_dataset = []
for tokens in positive_cleaned:
    positive_dataset.append(({token: True for token in tokens}, 'Positive')) # Navies bayes classier only takes dict comprehension

negative_dataset = []
for tokens in negative_cleaned:
    negative_dataset.append(({token: True for token in tokens}, 'Negative'))
"""Combined datasets and shuffled them"""
dataset = positive_dataset + negative_dataset
import random

random.shuffle(dataset)
random.shuffle(dataset)
random.shuffle(dataset)
random.shuffle(dataset)
"""takes only first 70% for training and last 30% for testing  """
training = dataset[:7000]
testing = dataset[7000:]
print(len(dataset),len(training),len(testing))

"""Pass data to classifer and get accuracy and most informative features"""
classifier = NaiveBayesClassifier.train(training)
print('Accuracy', classify.accuracy(classifier, testing))
print(classifier.show_most_informative_features(15)) #show me most influential tokens

from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
""" User Inferace to pass input to classifer to get a prediction"""
while True:
    response = input('Enter sentence').lower()
    tokens = word_tokenize(response)
    print(tokens)
    cleaned_tokens = clean(tokens)
    print(cleaned_tokens)
    response_dataset = {token: True for token in cleaned_tokens} #result in front iterable in back

    result = classifier.classify(response_dataset)
    print(result)
    print(analyzer.polarity_scores(' '.join(cleaned_tokens)))