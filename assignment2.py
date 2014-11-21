import glob
import json
import nltk
from pprint import pprint

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

def main():
	# Just testing my functions a bit
	listOfFiles = listFiles()
	print (listOfFiles[0])
	f1 = loadJSONfile(listOfFiles[0])
	pprint(len(f1['sentences']))

if __name__ == '__main__':
	main()
