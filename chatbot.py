# Meet FinanceBot: Your Financial Assistant

# import necessary libraries
import io
import random
import string  # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Uncomment the following only the first time
# nltk.download('punkt')  # first-time use only
# nltk.download('wordnet')  # first-time use only

from nltk.stem import WordNetLemmatizer
import nltk

# Download popular nltk packages
nltk.download('popular', quiet=True)

# Reading in the finance corpus
with open('finance_corpus.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

# Tokenization
sent_tokens = nltk.sent_tokenize(raw)  # converts to a list of sentences
word_tokens = nltk.word_tokenize(raw)  # converts to a list of words

# Preprocessing
lemmer = WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there",
                      "hello", "I am glad! You are talking to me"]


def greeting(sentence):
    """If the user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Generating response
def response(user_response):
    finance_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        finance_response = finance_response + \
            "I am sorry! I don't really understand you right now."
        return finance_response
    else:
        finance_response = finance_response + sent_tokens[idx]
        return finance_response


flag = True
print("FinanceBot: Hello! I'm FinanceBot, your financial assistant. I can help you with questions about finance. If you want to exit, type 'BYE.'")
while flag:
    user_response = input()
    user_response = user_response.lower()
    if user_response != 'BYE! ':
        if user_response == 'Thanks' or user_response == 'Thank You!':
            flag = False
            print("FinanceBot: You are more than welcome!")
        else:
            if greeting(user_response) is not None:
                print("FinanceBot: " + greeting(user_response))
            else:
                print("FinanceBot: ", end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("FinanceBot: Goodbye! Hope to assist you again.")
