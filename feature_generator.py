import xml.etree.ElementTree as ET
import textacy
import string
import nltk
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

def pronouns_conjunctions(tree):
    tokenize = lambda doc: doc.lower().split()
    first_person = ["i", "me", "we", "us"]
    second_person = ["you"]
    third_person = ["she", "her", "him", "it", "they", "them"]
    f, s, t, conjunction_count, modal_verb_count = 0,0,0,0,0
    try:
        text = tree.find("mainText").text
        text = "".join([ch for ch in text if ch not in string.punctuation])

        list_of_words_tags = nltk.pos_tag(tokenize(text))

        for word, tag in list_of_words_tags:
            if word in first_person: f += 1
            elif word in second_person: s += 1
            elif word in third_person: t += 1
            if tag == 'CC':
                conjunction_count += 1
            elif tag == 'MD':
                modal_verb_count += 1
    except Exception as e:
        print(e)
    return [f, s, t, conjunction_count, modal_verb_count]

"""
Functions for uncertainty counting uncertainty words (hedge and weasel words), and anything else using a gazetteer.
For the source of the word dictionaries I use, please see:
https://github.com/words/hedges
https://github.com/words/weasels
"""

def get_gazetteer(filename):
    """Takes a text file and returns a list object, with each row as an element"""
    content = None # Intialize
    with open(filename) as f:
        content = f.readlines()
    content = [line.strip() for line in content] # Remove newlines from the words
    result = []
    for word in content: # Strip any string that contains a substring - these throw off the overall word count
        if not any([r in word for r in result if word != r]):
            result.append(word)
    return result

def get_word_count(text,list_of_words): 
    """Takes list of words and text, returns # of occurences of each word from the list in the text"""
    count = 0
    for word in hedge_words: 
        count += text.lower().count(word)
    return count

def get_number_words(tree,gazetteer):
    """
    Takes a dictionary txt file and text, returns count of the file words in the text.
    Created for the hedge and weasel word features.  
    """
    text = ''
    if tree.find("mainText").text:
        text = tree.find("mainText").text
    count = 0
    if text == '': # Return count = 0 if there is no mainText (precaution)
        return count
    else: 
        count = get_word_count(text,gazetteer)
        return count


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

        # number of links
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
            tag, value = "number_of_sentences", str(text_stats.n_sents)
            tree = add_element(tree, tag, value)
            tag, value = "number_of_long_words", str(text_stats.n_long_words)
            tree = add_element(tree, tag, value)
            tag, value = "number_of_monosyllable_words", str(text_stats.n_monosyllable_words)
            tree = add_element(tree, tag, value)
            tag, value = "number_of_polsyllable_words", str(text_stats.n_polysyllable_words)
            tree = add_element(tree, tag, value)
            avg_no_of_syllables = text_stats.n_syllables / text_stats.n_sents
            tag, value = "average_number_of_syllables", str(avg_no_of_syllables)
            tree = add_element(tree, tag, value)
            tag, value = "flesch_readability_ease", str(text_stats.flesch_readability_ease)
            tree = add_element(tree, tag, value)

            # pronouns, conjunctions, modal verbs
            first, second, third, conjunction_count, modal_verb_count = pronouns_conjunctions(tree)
            tag, value = "first_person_pronouns", str(first)
            tree = add_element(tree, tag, value)
            tag, value = "second_person_pronouns", str(second)
            tree = add_element(tree, tag, value)
            tag, value = "third_person_pronouns", str(third)
            tree = add_element(tree, tag, value)
            tag, value = "conjunction_count", str(conjunction_count)
            tree = add_element(tree, tag, value)
            tag, value = "modal_verb_count", str(modal_verb_count)
            tree = add_element(tree, tag, value)

            # Uncertainty words
            # tag, value = "number_of_hedge_words", str(avg_no_of_syllables)
            # tag, value = "number_of_weasel_words", str(avg_no_of_syllables)

            #To do
            """
            3)Grammatical Complexity (number of short
            sentences, number of long sentences, FleshKincaid
            grade level, average number of
            words per sentence, sentence complexity,
            number of conjunctions),
            4) Uncertainty (Number of words express certainty,
            number of tentative words, modal
            verbs)
            5) Specificity and Expressiveness (rate of adjectives
            and adverbs, number of affective terms)
            """
        tree.write(xmlfile)

if __name__ == '__main__':
    add_features()
