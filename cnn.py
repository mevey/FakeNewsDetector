from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten
from parser import *
import numpy as np
import os
# fix random seed for reproducibility
np.random.seed(7)

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
sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
sklearn_representation = sklearn_tfidf.fit_transform(documents)

X_train, X_test, y_train, y_test = train_test_split(sklearn_representation, predictions, test_size = 0, random_state=25)


X = np.concatenate((X_train.todense() ,features), axis=1)
Y = np.reshape(y_train, (len(y_train), ))
dim = X.shape

# model
model = Sequential()
model.add(Dense(12, input_dim=dim[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=70, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
