import xml.etree.ElementTree as ET
import numpy as np
import os

def read_files(cols):
    """
    For each xml file return a matrix of values asked for
    """
    path = 'data/train/'
    possibilities = ['mixture of true and false', 'mostly false', 'no factual content', 'mostly true']
    for filename in os.listdir(path):
        data_row = []
        if not filename.endswith('.xml'): continue
        xmlfile = os.path.join(path, filename)
        tree = ET.parse(xmlfile)
        if cols == "mainText":
            if tree.find("mainText").text:
                yield tree.find("mainText").text
            else:
                yield ''
        elif cols == "veracity":
            v = possibilities.index(tree.find("veracity").text)
            yield v
        else:
            for col in cols:
                try:
                    data_row.append(int(tree.find(col).text))
                except:
                    data_row.append(0)
            yield data_row


def feature_matrix(cols):
    data = []
    for row in read_files(cols):
        data.append(row)
    return np.array(data)

def document_text():
    data = []
    for row in read_files("mainText"):
        data.append(row)
    return data

def veracity():
    data = []
    for row in read_files("veracity"):
        data.append(row)
    return data