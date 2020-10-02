#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:33:49 2020

@author: siddharthsmac
"""

import pandas as pd
import re

df = pd.read_csv('/users/siddharthsmac/downloads/fake-news/train.csv')

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

df = df.dropna()

messages = df.copy()

messages.reset_index(inplace = True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()

corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
cv = CountVectorizer(ngram_range =  (1,3), max_features = 5000)

X = cv.fit_transform(corpus).toarray()
y = messages['label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

countvector_df = pd.DataFrame(X_train, columns = cv.get_feature_names())

import matplotlib.pyplot as plt


## Don't remember the code ....
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
   
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    
from sklearn.naive_bayes import MultinomialNB

classifer = MultinomialNB()

import numpy as np
import itertools
from sklearn import metrics

classifer.fit(X_train, y_train)
 
pred = classifer.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print('ccuraccy: %0.3f'%score) 
cm = metrics.confusion_matrix(y_test, pred) 
plot_confusion_matrix(cm, classes = ['FAKE', 'REAL'])  


## Passive Agressive Classifer Algorithm

from sklearn.linear_model import PassiveAggressiveClassifier

linear_clf = PassiveAggressiveClassifier()


linear_clf.fit(X_train, y_train)

pred = linear_clf.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print('ccuraccy: %0.3f'%score) 
cm = metrics.confusion_matrix(y_test, pred) 
plot_confusion_matrix(cm, classes = ['FAKE DATA', 'REAL DATA'])  


# Multinomial Classifer with Hyperparameter tuning

classifier = MultinomialNB(alpha = 0.1)

previous_score = 0
for alpha in np.arange(0, 2 , 0.1):
    sub_classifier = MultinomialNB(alpha = alpha)
    sub_classifier.fit(X_train, y_train)
    y_pred = sub_classifier.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    if score > previous_score:
        classifier = sub_classifier
    print('alpha: {}, score: {}'.format(alpha, score))

## Real Vs Fake Words

feature_names = cv.get_feature_names()

classifier.coef_[0]

# Most Real Words
print(sorted(zip(classifier.coef_[0], feature_names), reverse = True)[:30])

# Most Fake Words
print(sorted(zip(classifier.coef_[0], feature_names))[:5000])
