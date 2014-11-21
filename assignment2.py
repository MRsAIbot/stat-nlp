import glob
import json
import nltk
import pickle # Not used yet, but might be useful later
from collections import defaultdict
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

# Returns a dictionary with a count of all triggers
def getAllTriggers(fileList):
	triggerDict = defaultdict(int)
	for f in fileList:
		fJSON = loadJSONfile(f)
		for i in range(len(fJSON['sentences'])):
			triggerList = fJSON['sentences'][i]['eventCandidates']
			for trigger in triggerList:
				triggerDict[trigger['gold']] += 1

	return triggerDict

def main():
	# Just testing my functions a bit
	listOfFiles = listFiles()
	print (listOfFiles[0])
	f1 = loadJSONfile(listOfFiles[0])
	pprint(len(f1['sentences']))

	t = getAllTriggers(listOfFiles)

	pprint(t)


if __name__ == '__main__':
	main()
