import xml.etree.ElementTree as ET
import os

def read_files():
    """
    For each xml file return the main text and veracity
    """
    path = 'data/'
    for filename in os.listdir(path):
        if not filename.endswith('.xml'): continue
        xmlfile = os.path.join(path, filename)
        tree = ET.parse(xmlfile)
        yield (tree.find('mainText').text, tree.find('veracity').text)

