import glob
import json
import nltk
import pickle # Not used yet, but might be useful later
from collections import defaultdict
from pprint import pprint
import feature_vector
import numpy as np
import random
from scipy.sparse import vstack
from sklearn.naive_bayes import BernoulliNB
import naivebayes2 as nb
import utils
import warnings

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


#generate one training batch in perceptron algorithm for event triggers. 
#output: For all events in file file_name: the features (matrix) & triggers
def build_trigger_data_batch(file_name, FV, clf):
	trigger_list = []
	token_index_list = []
	sentence_list = []
	f_json = json.load(open(file_name))

	for sentence in f_json['sentences']:
		event_candidates_list = sentence['eventCandidates']
		for event in event_candidates_list:
			token_index_list.append( event['begin'] )
			sentence_list.append(sentence)
			trigger_list += [ event['gold'] ]

	matrix_list = []
	for token_index,sentence in zip(token_index_list, sentence_list):
		matrix_list.append( FV.get_feature_matrix(token_index, sentence, clf) )

	if clf=='perc':
		return matrix_list, trigger_list
	elif clf=='nb':
		return vstack(matrix_list), trigger_list	
            

#generate one training batch in perceptron algorithm for argument labels. 
#output: For all argument candidates in file file_name: 
#the features (matrix) & gold label of the trigger-argument relation
def build_argument_data_batch(file_name, FV, clf):
	gold_list = []
	matrix_list = []
	f_json = json.load(open(file_name))    
	for sentence in f_json['sentences']:
		event_candidates_list = sentence['eventCandidates']
		for event in event_candidates_list:
			argumentslist = event['arguments']
			for argument in argumentslist:
				arg_index = argument['begin']
				token_index = event['begin'] 
				matrix_list.append( FV.get_feature_matrix_argument_prediction(token_index, arg_index, sentence, clf) )
				gold_list.append( argument['gold'] )
	if clf=='perc':
		return matrix_list, gold_list
	elif clf=='nb':
		return vstack(matrix_list), gold_list


def build_dataset(file_list, FV, mode='trig', clf='nb'):
	"""
	This function construct the data matrix X and target vector y.

	Arguments:
	- clf: string -> 'nb' for naivebayes, 'perc' for perceptron

	Output:
	- X: data matrix which depends per type of classifier specified by mode
	- y: vector of classes
	"""
	if clf == 'nb':
		for file_index, file_name in enumerate(file_list):
			print 'Building test data from json file ',file_index , 'of', len(file_list)
			if mode == 'trig':
				if file_index == 0:
					X, y = build_trigger_data_batch(file_name, FV, clf='nb')
				else:
					(new_features, new_gold) = build_trigger_data_batch(file_name, FV, clf='nb')
					X = vstack((X,new_features))
					y += new_gold
			if mode == 'arg':
				if file_index == 0:
					X, y = build_argument_data_batch(file_name, FV, clf='nb')
				else:
					(new_features, new_gold) = build_argument_data_batch(file_name, FV, clf='nb')
					X = vstack((X,new_features))
					y += new_gold
			else:
				warnings.warn('Error in build_dataset: Must have mode "Trigger" or "Argument"!')
	elif clf == 'perc':
		X = []
		y = []
		for file_index, file_name in enumerate(file_list):
			print 'Building test data from json file ',file_index , 'of', len(file_list)
			if mode == 'trig':
				(feat_list_one_file, gold_list_one_file) = build_trigger_data_batch(filename, FV, clf='perc')
			elif mode == 'arg':
				(feat_list_one_file, gold_list_one_file) = build_argument_data_batch(filename, FV, clf='perc')
			else:
				warnings.warn('Error in build_dataset: Must have mode "Trigger" or "Argument"!' )
			X += feat_list_one_file
			y += gold_list_one_file
	else:
		warnings.warn('Error in build_dataset: Must have clf "nb" or "perc"!')

	return X,y


def main():
	# # Just testing my functions a bit
	# list_of_files = list_files()
	# print (list_of_files[0])
	# f1 = load_json_file(list_of_files[0])
	# pprint(len(f1['sentences']))
	    
	# # Finding and counting all event triggers
	# t = get_all_triggers(list_of_files)
	# print("Number of distinct event triggers: {0}".format(len(t.keys())))
	# pprint(t)

	# # Finding and counting all possible arguments (=relationship labels)
	# arg = get_all_arguments(list_of_files)
	# print("Number of relation arguments: {0}".format(len(arg.keys())))
	# pprint(arg)

	# # Test of sparse vectors
	# list_a = []
	# list_a.append(feature_vector.phi_alternative_0)
	# list_a.append(feature_vector.phi_alternative_1)

	# # listOfFiles = list_files()
	# f1 = load_json_file(list_of_files[0])
	# sentence = f1['sentences'][0]   #pick first sentence
	# token_index = 0 #first word in sentence

	# grammar_dict = feature_vector.identify_all_grammar_tags(list_of_files)   
	# all_grammar_tags = grammar_dict.keys()  #these lists should be saved and later loaded.

	# f_v=feature_vector.FeatureVector(list_a)
	# vec = f_v.get_vector_alternative( token_index, sentence, all_grammar_tags)

	# if 1:
	# 	f_matrix = f_v.get_feature_batch( [0,1,2], [sentence]*3, all_grammar_tags)
	# 	print np.array(f_matrix.todense())
	# 	print f_matrix.col
	# 	print f_matrix.row
	# 	print f_matrix.data

	# Read data
	FV_trig = feature_vector.FeatureVector('Trigger')
	train_list, valid_list = utils.create_training_and_validation_file_lists()
	print train_list
	print valid_list

	# # Test Naive Bayes
	# X = np.random.randint(2, size=(20,100))
	# print X
	# y = np.array([t.keys()[i] for i in [random.randint(0,9) for p in range(20)]])
	# print y
	# # a,b = X[2:3].shape
	# # print a
	# # print b

	# NB = nb.NaiveBayes()
	# NB.train(X,y)
	# print(NB.predict(X))
	# CM, prec, rec, F1 = NB.evaluate(X,y)
	# print prec
	# print rec
	# print F1
	# print CM

	# clf = BernoulliNB()
	# clf.fit(X,y)
	# print(clf.predict(X))


if __name__ == '__main__':
    main()
