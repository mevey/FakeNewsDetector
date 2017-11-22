from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize, max_features=10000)
sklearn_representation = sklearn_tfidf.fit_transform(documents)

#input_ = sklearn_representation
input_ = np.concatenate((sklearn_representation.todense() ,features), axis=1) #possible to array

# Splits data into training and test
X_train, X_test, y_train, y_test = train_test_split(input_, predictions, test_size = .3, random_state=25)

#GridSearchCV
params = {'C':[.00001,.0001,.001,.01,.1,.2,.5,.8,1,2,5,10,100]} # Dict of values to search over for regularization strength
logreg = LogisticRegression(penalty='l2')
best_logreg = GridSearchCV(logreg, params, scoring="f1_micro")
best_logreg.fit(X_train,y_train)
print(best_logreg.best_score_)
logreg = best_logreg.best_estimator_
print(logreg)
y_pred = logreg.predict(X_test)
print(y_pred)

#Evaluate the model
"""
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
classification_report(y_test, y_pred))
"""

classification_report(y_test, y_pred)
