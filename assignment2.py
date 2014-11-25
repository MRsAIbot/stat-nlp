import glob
import json
import nltk
from pprint import pprint
import feature_vector

"""
TO DO
- creat document classes
- implement Naive Bayes
"""

# Create a list of .json file names
def listFiles(path="./bionlp2011genia-train-clean/*.json"):
	return glob.glob(path)

# Opens and loads a single json file name and returns it
def loadJSONfile(fileName):
	with open(fileName) as rawJSON:
		d = json.load(rawJSON)
		return d

# Instantiating the feature vector class (creating a dict of feature vectors)
f_v = feature_vector.FeatureVector(feature_vector.list_a)

def main():
	# Just testing my functions a bit
	listOfFiles = listFiles()
	print (listOfFiles[0])
	f1 = loadJSONfile(listOfFiles[0])
	pprint(len(f1['sentences']))
 
	# Testing code Feature Vector functionality
	print(f_v.get_vector(3,4))


if __name__ == '__main__':
	main()
