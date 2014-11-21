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
def list_files(path="./bionlp2011genia-train-clean/*.json"):
	return glob.glob(path)

# Opens and loads a single json file name and returns it
def load_json_file(file_name):
	with open(file_name) as raw_json:
		d = json.load(raw_json)
		return d

# Returns a dictionary with a count of all triggers
def get_all_triggers(file_list):
	trigger_dict = defaultdict(int)
	for f in file_list:
		f_json = load_json_file(f)
		for i in range(len(f_json['sentences'])):
			trigger_list = f_json['sentences'][i]['eventCandidates']
			for trigger in trigger_list:
				trigger_dict[trigger['gold']] += 1

	return trigger_dict

def main():
	# Just testing my functions a bit
	list_of_files = list_files()
	print (list_of_files[0])
	f1 = load_json_file(list_of_files[0])
	pprint(len(f1['sentences']))

	# Finding and counting all event triggers
	t = get_all_triggers(list_of_files)
	print("Number of distinct event triggers: {0}".format(len(t.keys())))
	pprint(t)


if __name__ == '__main__':
	main()
