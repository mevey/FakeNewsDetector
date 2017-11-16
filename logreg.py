from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import xml.etree.ElementTree as ET
from parser import *
import os
from gensim.models import Word2Vec

documents = get_document_text()
predictions = get_veracity()
features = feature_matrix([
    "number_of_quotes",
    "number_of_links"
])

#Calculate TF-IDF over the main text of each article, creating vector representations of them
tokenize = lambda doc: doc.lower().split(" ")
sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
sklearn_representation = sklearn_tfidf.fit_transform(documents)

# Splits data into training and test
X_train, X_test, y_train, y_test = train_test_split(sklearn_representation, predictions, test_size = .3, random_state=25)
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
y_pred = LogReg.predict(X_test)
print(y_pred)

#Evaluate the model
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(classification_report(y_test, y_pred))
