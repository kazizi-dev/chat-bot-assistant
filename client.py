"""
Creates the GUI for interacting with the bot using tkinter.
"""

import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import json
import numpy as np
import pickle
import random
from tkinter import *


model = load_model('./results/chatbot_model.h5')
intents = json.loads(open('./data/intents.json', encoding='utf-8').read())
words = pickle.load(open('./results/words.pkl', 'rb'))
tags = pickle.load(open('./results/classes.pkl', 'rb'))


def clean_up_sentence(sentence):
    """
    cleans up any sentence that is inputted.
    """
    lemmatizer = WordNetLemmatizer()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, show_details=True):
    """
    takes sentences that are cleaned up and creates
    a collection of words out of them for predicting
    tags.
    """
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))


def predict_class(sentence):
    """
    outputs a list of intents, the probabilities,
    their likelihood of matching the correct intent.
    """
    # filter out predictions below a threshold
    p = bow(sentence, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25      # threshold to avoid overfitting
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": tags[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    """
    takes the list outputted and checks the json file
    and outputs the most response with the highest prob.
    """
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    """
    takes in the user's message from the GUI, then
    predicts the class and outputs the response.
    """
    ints = predict_class(msg)
    res = getResponse(ints, intents)
    return res


def send(self):
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#A6192E", font=("Verdana", 12 ))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


if __name__ == "__main__":
    base = Tk()
    base.title("Chatbot Assistant")
    base.geometry("400x500")
    base.resizable(width=FALSE, height=FALSE)

    # Create Chat window
    ChatLog = Text(base, bd=0, bg="#ffffff", height="8", width="50", font="Arial", fg='#A6192E')

    ChatLog.config(state=DISABLED)

    # Bind scrollbar to Chat window
    scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
    ChatLog['yscrollcommand'] = scrollbar.set

    # Create the box to enter message
    EntryBox = Text(base, bd=0, bg="#A6192E", width="29", height="5", font="Arial", fg='white')
    EntryBox.bind("<Return>", send)

    # Place all components on the screen
    scrollbar.place(x=376, y=6, height=386)
    ChatLog.place(x=6, y=6, height=386, width=370)
    EntryBox.place(x=6, y=401, height=90, width=370)
    # SendButton.place(x=6, y=401, height=90)

    base.mainloop()