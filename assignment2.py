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
f_v = feature_vector.FeatureVector()

def main():
	# Just testing my functions a bit
	listOfFiles = listFiles()
	print (listOfFiles[0])
	f1 = loadJSONfile(listOfFiles[0])
	pprint(len(f1['sentences']))

	# Testing code Feature Vector functionality
	f_v[('test','Regulation')] =1
	print (f_v)
	print (f_v['missing key'])

if __name__ == '__main__':
	main()
