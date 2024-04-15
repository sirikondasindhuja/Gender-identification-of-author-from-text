from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from CustomButton import TkinterCustomButton

from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import pickle
import os
from nltk.stem import PorterStemmer
from sklearn.preprocessing import StandardScaler

main = Tk()
main.title("GENDER IDENTIFICATION OF AUTHOR FROM TEXT")
main.geometry("1300x1200")

authors = ['EAP', 'HPL', 'MWS']

global document1, document2

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

gender_X = np.load("model/gender_X.npy")
gender_Y = np.load("model/gender_Y.npy")
indices = np.arange(gender_X.shape[0])
np.random.shuffle(indices)
gender_X = gender_X[indices]
gender_Y = gender_Y[indices]
normalize1 = StandardScaler()
gender_X = normalize1.fit_transform(gender_X)

author_X = np.load("model/author_X.npy")
author_Y = np.load("model/author_Y.npy")
indices = np.arange(author_X.shape[0])
np.random.shuffle(indices)
author_X = author_X[indices]
author_Y = author_Y[indices]
normalize2 = StandardScaler()
author_X = normalize2.fit_transform(author_X)

with open('model/gender_vector.txt', 'rb') as file:
    gender_vectorizer = pickle.load(file)
file.close()

with open('model/gender_rf.txt', 'rb') as file:
    gender_rf = pickle.load(file)
file.close()

with open('model/author_vector.txt', 'rb') as file:
    author_vectorizer = pickle.load(file)
file.close()

with open('model/author_rf.txt', 'rb') as file:
    author_rf = pickle.load(file)
file.close()

def getTextData(filename):
    file = open(filename,mode='r')
    textData = file.read()
    file.close()
    return textData

def uploadFirstDocument():
    global document1
    text.delete('1.0', END)
    document1 = askopenfilename(initialdir = "TextDocuments")
    tf1.insert(END,str(document1))
    text.insert(END,"First Document Loaded\n\n")
    document1 = getTextData(document1)
    text.insert(END,"First Document Text : "+document1+"\n\n")

def uploadSecondDocument():
    global document2
    text.delete('1.0', END)
    document2 = askopenfilename(initialdir = "TextDocuments")
    tf2.insert(END,str(document2))
    text.insert(END,"Second Document Loaded\n\n")
    document2 = getTextData(document2)
    text.insert(END,"Second Document Text : "+document2+"\n\n")
    
def getAuthorPrediction(textData):
    state = textData
    state = state.strip().lower()
    state = cleanPost(state)
    temp = []
    temp.append(state)
    temp = author_vectorizer.transform(temp).toarray()
    temp = normalize2.transform(temp)
    action = author_rf.predict(temp)
    return action[0]

def getGenderPrediction(textData):
    state = textData
    state = state.strip().lower()
    state = cleanPost(state)
    temp = []
    temp.append(state)
    temp = gender_vectorizer.transform(temp).toarray()
    temp = normalize1.transform(temp)
    action = gender_rf.predict(temp)
    return action[0]    

def profileAuthor():
    text.delete('1.0', END)
    global document1, document2
       
    doc1_author = getAuthorPrediction(document1)
    doc2_author = getAuthorPrediction(document2)
    if doc1_author == doc2_author:
        text.insert(END,"Both Documents Author is Same\n")
    else:
        text.insert(END,"Both Documents Author is Different\n")
        

def predictGender():
    global document1, document2
    doc1_gender = getGenderPrediction(document1+" "+document2)
    if doc1_gender == 0:
        text.insert(END,"Gender Detected from Text = Male\n")
    else:
        text.insert(END,"Gender Detected from Text = Female\n")
    
font = ('times', 15, 'bold')
title = Label(main, text='GENDER IDENTIFICATION OF AUTHOR FROM TEXT')
title.config(bg='HotPink4', fg='yellow2')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

l1 = Label(main, text='Document1 Location:')
l1.config(font=font1)
l1.place(x=50,y=100)

tf1 = Entry(main,width=60)
tf1.config(font=font1)
tf1.place(x=230,y=100)

upload1Button = TkinterCustomButton(text="Upload First Document", width=300, corner_radius=5, command=uploadFirstDocument)
upload1Button.place(x=780,y=90)

l2 = Label(main, text='Document2 Location:')
l2.config(font=font1)
l2.place(x=50,y=150)

tf2 = Entry(main,width=60)
tf2.config(font=font1)
tf2.place(x=230,y=150)

upload2Button = TkinterCustomButton(text="Upload Second Document", width=300, corner_radius=5, command=uploadSecondDocument)
upload2Button.place(x=780,y=140)

identifyButton = TkinterCustomButton(text="Predict Author Profiling", width=300, corner_radius=5, command=profileAuthor)
identifyButton.place(x=200,y=200)

identifyButton1 = TkinterCustomButton(text="Predict Author Gender", width=300, corner_radius=5, command=predictGender)
identifyButton1.place(x=550,y=200)


font1 = ('times', 13, 'bold')
text=Text(main,height=20,width=130)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)

main.config(bg='plum2')
main.mainloop()
