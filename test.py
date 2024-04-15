from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

textdata = []
labels = []

authors = ['EAP', 'HPL', 'MWS']

def getLabel(name):
    label = 0
    for i in range(len(authors)):
        if authors[i] == name:
            label = i
            break
    return label    

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

dataset = pd.read_csv("GenderDataset/GenderData.csv", encoding='iso-8859-1')
for i in range(len(dataset)):
    msg = dataset.get_value(i, 'Text')
    label = dataset.get_value(i, 'Gender')
    label = int(label)
    msg = msg.strip().lower()
    clean = cleanPost(msg)
    textdata.append(clean)
    labels.append(label)
    print(str(i)+" "+str(label))



textdata = np.asarray(textdata)
labels = np.asarray(labels)

vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace')
wordembed = vectorizer.fit_transform(textdata).toarray()
np.save("model/gender_X", wordembed)
np.save("model/gender_Y", labels)
with open('model/gender_vector.txt', 'wb') as file:
    pickle.dump(vectorizer, file)
file.close()

pca = PCA(n_components=500)
X = pca.fit_transform(X)

with open('model/gender_pca.txt', 'wb') as file:
    pickle.dump(pca, file)
file.close()

X = np.load("model/gender_X.npy")
Y = np.load("model/gender_Y.npy")
print(X.shape)
print(Y)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
print(np.unique(Y, return_counts=True))
normalize = StandardScaler()
X = normalize.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
X_train, X_test1, y_train, y_test1 = train_test_split(X, Y, test_size=0.1)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
predict = rf.predict(X_test)
acc = accuracy_score(predict, y_test)
print(acc)

with open('model/gender_rf.txt', 'wb') as file:
    pickle.dump(rf, file)
file.close()

with open('model/gender_vector.txt', 'rb') as file:
    vectorizer = pickle.load(file)
file.close()

with open('model/gender_rf.txt', 'rb') as file:
    rf = pickle.load(file)
file.close()





