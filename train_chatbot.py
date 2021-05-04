"""
Reads the natural language data, and puts it in a training set.
Then, it uses Keras sequential neural network to create a model.
"""

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

def perform_ETL():
    data_file = open('intents.json', encoding='utf-8').read()
    intents = json.loads(data_file)

    words=[]
    tags = []
    documents = []
    ignore_words = ['?', '!']

    # extract data from intents
    for intent in intents['intents']:
        for question in intent['questions']:
            words_list = nltk.word_tokenize(question)
            words.extend(words_list)

            documents.append((words_list, intent['tag']))

            if intent['tag'] not in tags:
                tags.append(intent['tag'])

    # lemmatize the words
    temp = []
    for w in words:
        if w not in ignore_words:
            temp.append(lemmatizer.lemmatize(w.lower()))

    words = sorted(list(set(temp)))
    tags = sorted(list(set(tags)))

    return documents, tags, words


if __name__ == "__main__":
    documents, tags, words = perform_ETL()

    print (len(documents), "documents")
    print (len(tags), "classe(s)", tags)
    print (len(words), "unique lemmatized words", words)

    # save the data as pickle files
    pickle.dump(words,open('./results/words.pkl','wb'))
    pickle.dump(tags,open('./results/classes.pkl','wb'))