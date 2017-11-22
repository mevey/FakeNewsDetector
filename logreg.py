from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score
import xml.etree.ElementTree as ET
from parser import *
import os
from gensim.models import Word2Vec

documents = get_document_text()
predictions = get_veracity()
features = feature_matrix([
    "number_of_quotes",
    "number_of_links",
    "number_of_words",
    "number_of_unique_words",
    "number_of_long_words",
    "number_of_monosyllable_words",
    "number_of_polsyllable_words",
    "number_of_syllables",
    "flesch_readability_ease"
])

#Calculate TF-IDF over the main text of each article, creating vector representations of them
tokenize = lambda doc: doc.lower().split(" ")
sklearn_tfidf = TfidfVectorizer(norm='l2', sublinear_tf=True, tokenizer=tokenize, max_features=1000)
sklearn_representation = sklearn_tfidf.fit_transform(documents)
print("Done vectorizing")

#input_ = sklearn_representation
#input_ = np.concatenate((sklearn_representation.toarray() ,features), axis=1)
input_ = sklearn_representation

# Splits data into training and test
X_train, X_test, y_train, y_test = train_test_split(input_, predictions, test_size = .3, random_state=25)

logreg = LogisticRegressionCV(penalty='l2', scoring="roc_auc")
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
print(y_pred)
print(logreg.score(X_test, y_test))

#Evaluate the model
"""
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
classification_report(y_test, y_pred))
"""

#print(classification_report(y_test, y_pred))
