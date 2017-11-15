import xml.etree.ElementTree as ET
import os

def write_to_file(xmlfile, tag, value):
    """
    Function adds or updates new elements to the file
    :param xmlfile:
    :param tag:
    :param value:
    :return: None. Updates the data file with the new element and value
    """
    tree = ET.parse(xmlfile)
    if tree.findall(tag):
        new_tag = tree.find(tag)
    else:
        new_tag = ET.SubElement(tree.getroot(), tag)
    new_tag.text = value
    tree.write(xmlfile)

########################################################################################################################
# Add Features Functions here
########################################################################################################################

def number_of_qoutes(tree):
    """ return the number of quotes in the file """
    return len(tree.findall('quotes'))

def number_of_links(tree):
    """ return the number of quotes in the file """
    return len(tree.findall('links'))


########################################################################################################################

#######################################################################################################################