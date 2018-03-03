# How to setup FakeNewsDetector

pip install -r requirements.txt

python -m spacy download en

python feature_generator.py


# Note on the data:
* The data were borrowed from another researcher and thus are not hosted on the public Github repo. Please contact us and we can request permission to share the data. 

# Purpose of each file:

## test_validation.py:
* Creates and runs the entire model, including both logistic regressions, the CNN and the RNN
* This includes training the models and evaluating them on the final test set

## feature_characteristics.tsv
* Contains statistics on how the various features vary across the different veracity labels

## feature_generator.py
* Contains functions for making features across the dataset (which is a collection XML files)

## model.ipynb
* Notebook containing a quick runthrough of our model, including some notes on how we implemented the various steps.
* Mostly used for testing purposes, this is not meant to be a full running version of the model. See test_validation.py for full model 

## parser.py
* Contains functions for reading through the XML data, writing in feature values and getting statistics on feature values


