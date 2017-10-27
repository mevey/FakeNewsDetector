from sklearn.feature_extraction.text import TfidfVectorizer
import xml.etree.ElementTree as ET
import os

#splits each document into individual words
tokenize = lambda doc: doc.lower().split(" ")

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


documents = [f[0] for f in read_files() if f[0] is not None]

sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
sklearn_representation = sklearn_tfidf.fit_transform(documents)

print(sklearn_representation)
