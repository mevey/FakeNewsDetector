import xml.etree.ElementTree as ET
import english_syllable
import textacy
import os

def add_element(tree, tag, value):
    """
    Function adds or updates new tags. if tag already exists update it, if not, create the tag and add it's value
    :param tree:
    :param tag:
    :param value:
    :return: None. Updates the data file with the new element and value
    """
    if tree.findall(tag):
        new_tag = tree.find(tag)
    else:
        new_tag = ET.SubElement(tree.getroot(), tag)
    new_tag.text = value
    return tree


########################################################################################################################
# Add Features Functions here
########################################################################################################################

def number_of_qoutes(tree):
    """ return the number of quotes in the file """
    return len(tree.findall('quotes'))

def number_of_links(tree):
    """ return the number of links in the file """
    return len(tree.findall('links'))

def contains_author(tree):
    """Returns true if an author exists else False"""
    return 1 if tree.findall('author') else 0

def word_stats(tree):
    """Returns a bunch of textacy stats on the text"""
    stats = None
    try:
        text = tree.find("mainText").text
        doc = textacy.Doc(text)
        stats = textacy.text_stats.TextStats(doc)
    except Exception as e:
        print(e)
    return stats

########################################################################################################################
#Add features (element, tag) to file
########################################################################################################################

def add_features():
    path = 'data/train/'
    for filename in os.listdir(path):
        if not filename.endswith('.xml'): continue
        xmlfile = os.path.join(path, filename)
        print("Updating %s" % xmlfile)
        try:
            tree = ET.parse(xmlfile)
        except:
            continue

        #All features should be number valued
        #number of quotes
        tag, value = "number_of_quotes", str(number_of_qoutes(tree))
        tree = add_element(tree, tag, value)

        #number of links
        tag, value = "number_of_links", str(number_of_links(tree))
        tree = add_element(tree, tag, value)

        # number of links
        tag, value = "contains_author", str(contains_author(tree))
        tree = add_element(tree, tag, value)

        text_stats = word_stats(tree)
        if text_stats:
            tag, value = "number_of_words", str(text_stats.n_words)
            tree = add_element(tree, tag, value)
            tag, value = "number_of_unique_words", str(text_stats.n_unique_words)
            tree = add_element(tree, tag, value)
            tag, value = "number_of_long_words", str(text_stats.n_long_words)
            tree = add_element(tree, tag, value)
            tag, value = "number_of_monosyllable_words", str(text_stats.n_monosyllable_words)
            tree = add_element(tree, tag, value)
            tag, value = "number_of_polsyllable_words", str(text_stats.n_polysyllable_words)
            tree = add_element(tree, tag, value)
            tag, value = "number_of_syllables", str(text_stats.n_syllables)
            tree = add_element(tree, tag, value)
            tag, value = "flesch_readability_ease", str(text_stats.flesch_readability_ease)
            tree = add_element(tree, tag, value)

        tree.write(xmlfile)

if __name__ == '__main__':
    add_features()
