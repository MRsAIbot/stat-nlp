import glob
import json
import nltk
import pickle # Not used yet, but might be useful later
from collections import defaultdict
from pprint import pprint
import feature_vector

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
 
#Returns a dictionary with a count of all arguments (=labels of the relations)
def get_all_arguments(file_list):
    argument_dict = defaultdict(int)
    for f in file_list:
        f_json = load_json_file(f)
        for sentence in f_json['sentences']:
            event_candidates_list = sentence['eventCandidates']
            for event_candidates in event_candidates_list:
                arguments_list = event_candidates['arguments']
                
                for argument in arguments_list:
                    argument_dict[argument['gold']] += 1
    return argument_dict
    

    
    

def main():
    # # Just testing my functions a bit
    # list_of_files = list_files()
    # print (list_of_files[0])
    # f1 = load_json_file(list_of_files[0])
    # pprint(len(f1['sentences']))
        
    # # Finding and counting all event triggers
    # # t = get_all_triggers(list_of_files)
    # # print("Number of distinct event triggers: {0}".format(len(t.keys())))
    # # pprint(t)
    
    # # Finding and counting all possible arguments (=relationship labels)
    # arg = get_all_arguments(list_of_files)
    # print("Number of relation arguments: {0}".format(len(arg.keys())))
    # pprint(arg)

    list_a = []
    list_a.append(feature_vector.phi_alternative_0)
    list_a.append(feature_vector.phi_alternative_1)

    if 1:
        listOfFiles = list_files()
        f1 = load_json_file(listOfFiles[0])
        sentence = f1['sentences'][0]   #pick first sentence
        token_index = 0 #first word in sentence

        grammar_dict = feature_vector.identify_all_grammar_tags(listOfFiles)   
        all_grammar_tags = grammar_dict.keys()  #these lists should be saved and later loaded.

        f_v=feature_vector.FeatureVector(list_a)
        vec = f_v.get_vector_alternative( token_index, sentence, all_grammar_tags)

        print all_grammar_tags
        print vec


if __name__ == '__main__':
    main()