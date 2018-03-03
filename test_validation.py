import numpy as np
np.random.seed(1337) # Fix a random seed to make reproducible results

#from parser import *


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV # Tools for splitting data, tuning hyperparameters
from sklearn.linear_model import LogisticRegressionCV # Logreg model
from sklearn.metrics import classification_report, accuracy_score, fbeta_score, make_scorer # Evaluation tools
from nltk.tokenize import word_tokenize # Tokenizer

import xml.etree.ElementTree as ET
import os
from gensim.models import Word2Vec, KeyedVectors

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Embedding, Input, Conv1D, MaxPooling1D, Dropout, LSTM
from keras.utils import to_categorical
from keras import backend as K
from keras.regularizers import l2
from keras.wrappers.scikit_learn import KerasClassifier # Wrapper to use Keras model in sklearn

########################################################################################################################
# Read in the data 
########################################################################################################################
possibilities = ['mixture of true and false', 'mostly false', 'no factual content', 'mostly true']
# possibilities = ['mixture of true and false', 'mostly false', 'mostly true']
def read_files(cols, orientation='all',data='all'):
    """
    For each xml file, return a matrix of values requested
    
    orientation: Only read out data for specific political orientations ('mainstream','')
    data: 'all' tests the full dataset, 'partisan' tests just the partisan articles (i.e. not mainstream/non-factual)
    """
    path = 'data/train/'
    for filename in os.listdir(path):
        data_row = []
        if not filename.endswith('.xml'): continue
        xmlfile = os.path.join(path, filename)
        tree = ET.parse(xmlfile)

# Testing whole dataset:        
        if data == 'all':
            if not tree.find("mainText").text: continue
            if orientation != "all" and tree.find("orientation").text != orientation:
                continue
        
# Testing only partisan dataset:
        elif data == 'partisan':
            if not tree.find("mainText").text or tree.find("veracity").text == "no factual content": continue
            if orientation == "all" and tree.find("orientation").text == 'mainstream':
                continue    

# For the dataset, yield the text, veracity label from XML files
        if cols == "mainText":
            if tree.find("mainText").text:
                yield tree.find("mainText").text
            else:
                continue
        elif cols == "veracity":
            v = possibilities.index(tree.find("veracity").text)
            yield v
        elif cols == "both":
            if tree.find("mainText").text:
                v = possibilities.index(tree.find("veracity").text)
                yield tree.find("mainText").text, v
            else:
                continue
        else:
            for col in cols:
                try:
                    data_row.append(float(tree.find(col).text))
                except:
                    data_row.append(0.0)
            yield data_row

def feature_matrix(cols):
    """Returns a matrix of the feature values for the files (note feature values have been written into XML)"""
    data = []
    for row in read_files(cols):
        data.append(np.array(row))
    return np.array(data)

def get_document_text():
    """Return list of article maintext - deprecated"""
    data = []
    for row in read_files("mainText"):
        if not row:
            continue
        else:
            data.append(row)
    return data

def get_veracity():
    """Return list of veracity labels - deprecated"""
    data = []
    for row in read_files("veracity"):
        data.append(row)
    return data

def get_document_text_and_veracity():
    # Return lists of document maintext and veracity labels
    docs, preds = [], []
    for row in read_files("both"):
        if not row[0]:
            continue
        else:
            docs.append(row[0])
            preds.append(row[1])
    return docs, preds

# documents, predictions = get_document_text_and_veracity()

# file = 'data/GoogleNews-vectors-negative300.bin'
# embeddings = KeyedVectors.load_word2vec_format(file, binary=True)

########################################################################################################################
# Define functions for processiong the logistic regression data
########################################################################################################################


def avg_docvec(docText,embeddings):
    """
    This function converts the text of a document (input as a string) to word embeddings, then
    takes the elementwise average of the embeddings to return a single vector.
    """
    docVec = np.zeros(300) # Initialize array for the document
    tokens = word_tokenize(docText) # Creates a list of word tokens (e.g. "Test words" -> ['Test', 'words'])
    denominator = 0.0 # To take the average, will only count tokens for which we have embeddings in the total  
    for token in tokens:
        try:
            v = embeddings[token]
            np.add(docVec,v,out=docVec)
            denominator += 1.0
        except: # Ignore tokens that aren't in the Google News embeddings
            continue
    np.divide(docVec,denominator,out=docVec) 
    return docVec

def max_docvec(docText,embeddings):
    """
    Converts the text of a document (input as a string) to word embeddings, then takes the elementwise
    max of the embeddings to return a single vector of the maximum elements.
    """
    docVec = 0
    tokens = word_tokenize(docText) # Creates a list of word tokens (e.g. "Test words" -> ['Test', 'words'])
    startIndex = 0
    for i in range(len(tokens)): # Initialize the doc vec as the first token that is in the embeddings
        try:
            v = embeddings[tokens[i]]
            docVec = v
            startIndex = i
            break
        except:
            continue
    
    for token in tokens[startIndex:]:
        try:
            v = embeddings[token]
            np.max(docVec,v,out=docVec)
        except: # Ignore tokens that aren't in the Google News embeddings
            continue
    return docVec

def min_docvec(docText,embeddings):
    """
    Converts the text of a document (input as a string) to word embeddings, then takes the elementwise
    min of the embeddings to return a single vector of the minimum elements.
    """
    docVec = 0
    tokens = word_tokenize(docText) # Creates a list of word tokens (e.g. "Test words" -> ['Test', 'words'])
    startIndex = 0
    for i in range(len(tokens)): # Initialize the doc vec as the first token that is in the embeddings
        try:
            v = embeddings[tokens[i]]
            docVec = v
            startIndex = i
            break
        except:
            continue
    for token in tokens[startIndex:]: # Loop over words in the article, starting at first valid word
        try:
            v = embeddings[token]
            np.min(docVec,v,out=docVec) # Only keep min elements
        except: # Ignore tokens that aren't in the Google News embeddings
            continue
    return docVec

def docs_to_matrix(documents,embeddings,method='avg'):
    """
    Takes a list of document text strings and returns a matrix of document embeddings.
    The method specifies how the word vectors are combined for the document: average is 
    element-wise average, min is element-wise min and max is element-wise max. 
    """
    matrix = []
    count = 0
    for i in range(len(documents)):
        vector = 0
        if method.lower() == 'avg':
            vector = avg_docvec(documents[i],embeddings)
        elif method.lower() == 'min':
            vector = min_docvec(documents[i],embeddings)
        elif method.lower() == 'max':
            vector = max_docvec(documents[i],embeddings)
        else:
            print("Please enter method argument as min, max or avg")
            return
        if i == 0:
            matrix = vector
        else:
            matrix = np.column_stack((matrix,vector)) # Concat all vectors into a matrix of order (300,N of docs)
            count += 1
    matrix = matrix.reshape((len(documents),300)) # For sklearn, reshape the matrix into order (N of docs,300), so rows = docs
    return matrix


########################################################################################################################
# first, we split the data and evaluate a logistic regression on it 
########################################################################################################################

TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.2

features = feature_matrix([
    "number_of_words",
    "number_of_unique_words",
    "number_of_sentences",
    "number_of_long_words",
    "number_of_monosyllable_words",
    "number_of_polsyllable_words",
    "average_number_of_syllables",
    "flesch_readability_ease",
    "first_person_pronouns",
    "second_person_pronouns",
    "third_person_pronouns",
    "conjunction_count",
    "modal_verb_count",
    "number_of_hedge_words",
    "number_of_weasel_words",
    "number_of_links",
    "number_of_quotes",
    "contains_author"
])



def run_logreg(documents,features,embeddings):
    """
    runs and evaluates logreg model
    Runs both embedding and tf-idf versions. 
    """

    # fbeta = make_scorer(fbeta_score,beta=5.0) experimented with fbeta. Not in use. 

    # Version 1: Use embeddings and concatenate all features, then do train-test split. 
    articles_matrix = docs_to_matrix(documents,embeddings)
    articles_matrix = np.concatenate((articles_matrix,features),axis=1)
    x_training_set, x_final_test, y_training_set, y_final_test = train_test_split(articles_matrix, predictions, test_size=TEST_SPLIT, random_state=25) 

    # Final Model test of the Logreg embeddings
    print("TEST RESULTS OF THE LOGREG WITH EMBEDDINGS")
    logreg = LogisticRegressionCV(penalty='l2', scoring='f1',Cs=[.00001,.0001,.001,.01,.1,.2,.5,.8,1,2,5,10,100,1000])
    logreg.fit(x_training_set,y_training_set)
    y_pred = logreg.predict(x_final_test)
    print(classification_report(y_final_test, y_pred))
    print(accuracy_score(y_final_test,y_pred))

    # Version 2: fit using the tf-idf features and all other features
    sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=word_tokenize)
    articles_matrix = sklearn_tfidf.fit_transform(documents)
    articles_matrix = np.concatenate((articles_matrix.toarray(),features),axis=1)
    x_training_set, x_final_test, y_training_set, y_final_test = train_test_split(articles_matrix, predictions, test_size=TEST_SPLIT, random_state=25) 

    # Final Model test of the Logreg tf-idf
    print("TEST RESULTS OF THE LOGREG WITH TF-IDF")
    logreg = LogisticRegressionCV(penalty='l2', scoring="f1",Cs=[.00001,.0001,.001,.01,.1,.2,.5,.8,1,2,5,10,100,1000])
    logreg.fit(x_training_set,y_training_set)
    y_pred = logreg.predict(x_final_test)
    print(classification_report(y_final_test, y_pred))
    print(accuracy_score(y_final_test,y_pred))


########################################################################################################################
# Create the CNN
########################################################################################################################

def prep_data(documents,predictions,embeddings,maxlen):
    """Preps output data for the neural models"""

    # Prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(documents)
    vocab_size = len(t.word_index) + 1

    # integer encode the documents
    encoded_docs = t.texts_to_sequences(documents)

    # pad our doc sequences to a max length of MAX_SEQUENCE_LENGTH words
    data = pad_sequences(encoded_docs, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    # Re-using the list of integer labels generated earlier, make a binary class matrix
    labels = to_categorical(np.asarray(predictions))

    # Split the full dataset into data used for training and the final test stage
    x_training_set, x_final_test, y_training_set, y_final_test = train_test_split(data, labels, test_size = TEST_SPLIT, random_state=25) 

    # Secondary split of training data into test and training
    x_train, x_dev, y_train, y_dev = train_test_split(x_training_set, y_training_set, test_size = VALIDATION_SPLIT, random_state=17)

    # create a weight matrix for words in training docs
    embedding_weights = np.zeros((vocab_size, 300))
    for word, i in t.word_index.items():
        embedding_vector = None
        try:
            embedding_vector = embeddings[word] # Get the google vector for a given word
        except:
            embedding_vector = None
        if embedding_vector is not None:
            embedding_weights[i] = embedding_vector

    x_data = {'train':x_train,'dev':x_dev,'test':x_final_test}
    y_data = {'train':y_train,'dev':y_dev,'test':y_final_test}

    return x_data, y_data, embedding_weights



def create_cnn(embedding_weights,embedding_dim,max_sequence_len,filters,k,dropout):    
    print("k= ",k," seq len= ",max_sequence_len," filters= ",filters," dropout= ",dropout)
    embedding_layer = Embedding(embedding_weights.shape[0], # This is the vocab size
                                EMBEDDING_DIM,
                                weights=[embedding_weights],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32') 
    embedded_sequences = embedding_layer(sequence_input) 

    x = Conv1D(FILTERS, k, activation='relu')(embedded_sequences) 
    x = MaxPooling1D(k)(x)
    x = Dropout(DROPOUT)(x)
    x = Conv1D(FILTERS, k, activation='relu')(x)
    x = MaxPooling1D(k)(x)
    x = Dropout(DROPOUT)(x)
    x = Conv1D(FILTERS, k, activation='relu')(x)
    x = MaxPooling1D(int(x.shape[1]))(x)  # This layer pools the entire previous layer
    x = Flatten()(x)
    x = Dense(FILTERS, activation='relu')(x)
    preds = Dense(len(possibilities), activation='softmax')(x)
    
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop', # rmsprop is the standard
                  metrics=['acc'])
    print(model.summary())
    return model


########################################################################################################################
# Bonus round - LSTM Model
########################################################################################################################

def create_rnn(embedding_weights,embedding_dim,max_sequence_len,filters,dropout):
    embedding_layer = Embedding(embedding_weights.shape[0],
                                embedding_dim,
                                weights=[embedding_weights],
                                input_length=max_sequence_len,
                                trainable=False)

    rnn = Sequential()
    rnn.add(embedding_layer)
    rnn.add(Dropout(0.5))
    rnn.add(LSTM(300))
    rnn.add(Dropout(0.5))
    rnn.add(Dense(len(possibilities), activation='sigmoid'))
    rnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(rnn.summary())
    return rnn

########################################################################################################################
# Run cross validation over the hyperparameters
########################################################################################################################

# TSB - Commenting this out now. Takes forever to run, should use random search next time


# filters_dev = [100,200,300]
# sequence_len_dev = [400,600,700,800]
# dropout_dev = [0.01,0.1,0.5,0.9,1.0]
# k_dev = [3,4,5,6]
# epochs_dev = [1,2,3] # No change across epoch values
# batch_size_dev = [16,32,64,128,256,300]
# param_grid = dict(max_sequence_len=sequence_len_dev,filters=filters_dev,k=k_dev,dropout=dropout_dev) 
# param_grid = dict(epochs=epochs_dev,batch_size=batch_size_dev)
# param_grid = dict(max_sequence_len=[500,600,700],k=[4,5,6],dropout=[0.5,0.9]) 

# # Cross-validation using sklearn's gridsearchCV (WARNING: very costly call)
# cnn = KerasClassifier(build_fn=create_cnn,embedding_weights=embedding_weights)
# grid = GridSearchCV(estimator=cnn,param_grid=param_grid,n_jobs=-1)
# grid_result = grid.fit(x_train,y_train) 

# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))



########################################################################################################################
# Final evaluation steps
########################################################################################################################

def make_classifications_list(binary_targets):
    """This is  for turning the output of Keras models into a list of class integers"""
    y_true = list(binary_targets)
    for i in range(len(y_true)):
        classification = None
        for j in range(y_true[i].shape[0]):
            if y_true[i][j] == 1.:
                classification = j
        y_true[i] = classification
    return y_true


# # Final Model Test - CNN
# model = create_cnn(embedding_weights,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH,FILTERS,k,DROPOUT)
# model.fit(x_train, y_train,epochs=1, batch_size=batch_size)
# y_prob = model.predict(x_final_test,batch_size=batch_size)
# y_pred = y_prob.argmax(axis=-1) # Get the predicted class (not probabilites of each)
# ypred = list(y_pred) # Turn array into list
# y_true = make_classifications_list(y_final_test) # Turn Matrix of targets into list
# print(classification_report(y_true, y_pred))
# print(accuracy_score(y_true,y_pred))
# scores = model.evaluate(x_final_test,y_final_test,verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))

# # Final Model Test - LSTM
# model = create_rnn(embedding_weights,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH,FILTERS,DROPOUT)
# model.fit(x_train, y_train,epochs=1, batch_size=batch_size)
# y_prob = model.predict(x_final_test,batch_size=batch_size)
# y_pred = y_prob.argmax(axis=-1) # Get the predicted class (not probabilites of each)
# ypred = list(y_pred) # Turn array into list
# y_true = make_classifications_list(y_final_test) # Turn Matrix of targets into list
# print(classification_report(y_true, y_pred))
# print(accuracy_score(y_true,y_pred))
# scores = model.evaluate(x_final_test,y_final_test,verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))

# One-off dev test - CNN
# model = create_cnn(embedding_weights,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH,FILTERS,k,DROPOUT)
# model.fit(x_train, y_train,epochs=1, batch_size=batch_size,class_weight=class_weight)
# y_prob = model.predict(x_dev,batch_size=batch_size)
# y_pred = y_prob.argmax(axis=-1) # Get the predicted class (not probabilites of each)
# ypred = list(y_pred) # Turn array into list
# y_true = make_classifications_list(y_dev) # Turn Matrix of targets into list
# print(classification_report(y_true, y_pred))
# print(accuracy_score(y_true,y_pred))
# scores = model.evaluate(x_dev,y_dev,verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))

# One-off dev test - LSTM
# model = create_rnn(embedding_weights,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH,FILTERS,DROPOUT)
# model.fit(x_train, y_train,epochs=1, batch_size=batch_size,class_weight=class_weight)
# y_prob = model.predict(x_dev,batch_size=batch_size)
# y_pred = y_prob.argmax(axis=-1) # Get the predicted class (not probabilites of each)
# ypred = list(y_pred) # Turn array into list
# y_true = make_classifications_list(y_dev) # Turn Matrix of targets into list
# print(classification_report(y_true, y_pred))
# print(accuracy_score(y_true,y_pred))
# scores = model.evaluate(x_dev,y_dev,verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))


########################################################################################################################
# Note - these are dev tests used in model tuning. Remove? 
########################################################################################################################

if __name__ == '__main__':
    possibilities = ['mixture of true and false', 'mostly false', 'no factual content', 'mostly true']
    # possibilities = ['mixture of true and false', 'mostly false', 'mostly true'] # For partisan-only

    # Load in google embeddings and the document text/labels
    documents, predictions = get_document_text_and_veracity()
    file = 'data/GoogleNews-vectors-negative300.bin'
    embeddings = KeyedVectors.load_word2vec_format(file, binary=True)

    # Create a matrix of all the feature values for each article
    features = feature_matrix([
        "number_of_words",
        "number_of_unique_words",
        "number_of_sentences",
        "number_of_long_words",
        "number_of_monosyllable_words",
        "number_of_polsyllable_words",
        "average_number_of_syllables",
        "flesch_readability_ease",
        "first_person_pronouns",
        "second_person_pronouns",
        "third_person_pronouns",
        "conjunction_count",
        "modal_verb_count",
        "number_of_hedge_words",
        "number_of_weasel_words",
        "number_of_links",
        "number_of_quotes",
        "contains_author"
    ])

    TEST_SPLIT = 0.2
    VALIDATION_SPLIT = 0.2
    MAX_SEQUENCE_LENGTH = 700 # Determined experimentally so far. 
    EMBEDDING_DIM = 300 # Google News embeddings are 300 dimensional
    DROPOUT = 0.5 # Dropout strength 
    FILTERS = 300 # Number of filters in the convolutional layers
    k = 5 # Sliding k window size for convolutional layers
    class_weight = {0:3,1:3,2:1,3:1} # Class weighting attempt to deal with imbalance
    batch_size = 300

    # First execute logistic regression
    run_logreg(documents,features,embeddings)

    # Prepare train/test data and embedding weights for neural models
    x_data, y_data, embedding_weights = prep_data(documents,predictions,embeddings,MAX_SEQUENCE_LENGTH)

    # Final Model Test - CNN
    model = create_cnn(embedding_weights,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH,FILTERS,k,DROPOUT)
    model.fit(x_data['train'], y_data['train'],epochs=1, batch_size=batch_size)
    y_prob = model.predict(x_data['test'],batch_size=batch_size)
    y_pred = y_prob.argmax(axis=-1) # Get the predicted class (not probabilites of each)
    ypred = list(y_pred) # Turn array into list
    y_true = make_classifications_list(y_data['test']) # Turn Matrix of targets into list
    print(classification_report(y_true, y_pred))
    print(accuracy_score(y_true,y_pred))
    scores = model.evaluate(x_data['test'],y_data['test'],verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    # Final Model Test - LSTM
    model = create_rnn(embedding_weights,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH,FILTERS,DROPOUT)
    model.fit(x_data['train'], y_data['train'],epochs=1, batch_size=batch_size)
    y_prob = model.predict(x_data['test'],batch_size=batch_size)
    y_pred = y_prob.argmax(axis=-1) # Get the predicted class (not probabilites of each)
    ypred = list(y_pred) # Turn array into list
    y_true = make_classifications_list(y_data['test']) # Turn Matrix of targets into list
    print(classification_report(y_true, y_pred))
    print(accuracy_score(y_true,y_pred))
    scores = model.evaluate(x_data['test'],y_data['test'],verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))