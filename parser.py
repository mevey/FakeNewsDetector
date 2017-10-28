from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import xml.etree.ElementTree as ET
import os

def read_files():
    """
    For each xml file return the main text and veracity
    """
    path = 'data/train/'
    for filename in os.listdir(path):
        if not filename.endswith('.xml'): continue
        xmlfile = os.path.join(path, filename)
        tree = ET.parse(xmlfile)
        yield (tree.find('mainText').text, tree.find('veracity').text)

tokenize = lambda doc: doc.lower().split(" ")

documents = [f[0] for f in read_files() if f[0] is not None]
possibilities = ['mixture of true and false', 'mostly false', 'no factual content', 'mostly true']
predictions = [possibilities.index(f[1]) for f in read_files() if f[0] is not None]

#Calculating TDIDF with Scikitlearn is much faster
sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
sklearn_representation = sklearn_tfidf.fit_transform(documents)

#splits data into training and test
X_train, X_test, y_train, y_test = train_test_split(sklearn_representation, predictions, test_size = .3, random_state=25)
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
y_pred = LogReg.predict(X_test)
print(y_pred)

#Evaluate the model
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(classification_report(y_test, y_pred))
