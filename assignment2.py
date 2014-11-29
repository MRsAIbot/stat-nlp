import json
import nltk
import pickle # Not used yet, but might be useful later
from pprint import pprint
import feature_vector
import utils
import numpy as np


"""
TO DO
- creat document classes
- implement Naive Bayes
"""


def main():
	# Just testing my functions a bit
	list_of_files = utils.list_files()
	print (list_of_files[0])
	f1 = utils.load_json_file(list_of_files[0])
	pprint(len(f1['sentences']))
	    
	# Finding and counting all event triggers
	# t = get_all_triggers(list_of_files)
	# print("Number of distinct event triggers: {0}".format(len(t.keys())))
	# pprint(t)

	# Finding and counting all possible arguments (=relationship labels)
	arg = utils.get_all_arguments(list_of_files)
	print("Number of relation arguments: {0}".format(len(arg.keys())))
	pprint(arg)

	# Test of sparse vectors
	list_a = []
	list_a.append(feature_vector.phi_alternative_0)
	list_a.append(feature_vector.phi_alternative_1)

	listOfFiles = utils.list_files()
	f1 = utils.load_json_file(listOfFiles[0])
	sentence = f1['sentences'][0]   #pick first sentence
	token_index = 0 #first word in sentence

	grammar_dict = feature_vector.identify_all_grammar_tags(listOfFiles)   
	all_grammar_tags = grammar_dict.keys()  #these lists should be saved and later loaded.

	f_v=feature_vector.FeatureVector(list_a)
	vec = f_v.get_vector_alternative( token_index, sentence, all_grammar_tags)

	if 1:
         f_matrix = f_v.get_feature_batch( [0,1,2], [sentence]*3, all_grammar_tags)
         print np.array(f_matrix.todense())
         print f_matrix.col
         print f_matrix.row
         print f_matrix.data



if __name__ == '__main__':
    main()
