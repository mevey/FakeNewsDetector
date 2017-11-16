import xml.etree.ElementTree as ET
import numpy as np
import os

def read_files(cols):
    """
    For each xml file return a matrix of values asked for
    """
    path = 'data/train/'
    for filename in os.listdir(path):
        data_row = []
        if not filename.endswith('.xml'): continue
        xmlfile = os.path.join(path, filename)
        tree = ET.parse(xmlfile)
        for col in cols:
            try:
                data_row.append(tree.find(col).text)
            except:
                data_row.append(0)
        yield data_row


def data_matrix(cols):
    data = []
    for row in read_files(cols):
        data.append(row)
    return np.array(data)