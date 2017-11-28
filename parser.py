import xml.etree.ElementTree as ET
from feature_names import feature_names
from collections import Counter
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

def get_document_text():
    data = []
    for row in read_files("mainText"):
        data.append(row)
    return data

def get_veracity():
    data = []
    for row in read_files("veracity"):
        data.append(row)
    return data

def data_distribution(col):
    """
    Return the statistics for each feature
    """
    title, distribution = "" , ""
    path = 'data/train/'
    possibilities = ['mixture of true and false', 'mostly false', 'no factual content', 'mostly true']
    stats = [[],[],[],[]]
    for filename in os.listdir(path):
        if not filename.endswith('.xml'): continue
        xmlfile = os.path.join(path, filename)
        tree = ET.parse(xmlfile)
        v = possibilities.index(tree.find("veracity").text)
        try:
            stats[v].append(int(tree.find(col).text))
        except:
            stats[v].append(0)
    if len(col) < 30: col += ("." * (30 - len(col)))
    title = "\t".join([col, "docs", "max","min","mode", "mean"]) + "\n"
    print(title)
    for i,stat in enumerate(stats):
        mean = sum(stat) / len(stat)
        mode = Counter(stat).most_common(1)
        Y = possibilities[i]
        if len(Y) < 30: Y += ("." * (30-len(Y)))
        distribution += "\t".join([Y, str(len(stat)), str(max(stat)), str(min(stat)), str(mode), str(mean)]) + "\n"
        print(distribution)
    return title, distribution

def write_to_feature_distribution_file():
    with open("feature_characteristics.tsv", "w") as f:
        for feature in feature_names:
            title, distribution = data_distribution(feature)
            f.write(title)
            f.write(distribution)
            f.write("\n\n")

if __name__ == '__main__':
    write_to_feature_distribution_file()