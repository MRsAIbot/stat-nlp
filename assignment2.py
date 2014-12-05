import nltk
import cPickle
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


# generate one training batch in perceptron algorithm for event triggers. 
# output: For all events in file file_name: the features (matrix) & triggers
def build_trigger_data_batch(file_name, FV, clf):
	trigger_list = []
	token_index_list = []
	sentence_list = []
	f_json = utils.load_json_file(file_name)

	for sentence in f_json['sentences']:
		event_candidates_list = sentence['eventCandidates']
		for event in event_candidates_list:
			token_index_list.append( event['begin'] )
			sentence_list.append(sentence)
			trigger_list += [ event['gold'] ]

	matrix_list = []
	for token_index,sentence in zip(token_index_list, sentence_list):
		matrix_list.append( FV.get_feature_matrix(token_index, sentence, clf) )

	if len(matrix_list) == 0:
		return None, None
	
	if clf=='perc':
		return matrix_list, trigger_list
	elif clf=='nb':
		return vstack(matrix_list), trigger_list	

# generate one training batch in perceptron algorithm for argument labels. 
# output: For all argument candidates in file file_name: 
# the features (matrix) & gold label of the trigger-argument relation
def build_argument_data_batch(file_name, FV, clf):
	gold_list = []
	matrix_list = []
	f_json = utils.load_json_file(file_name)

	for sentence in f_json['sentences']:
		event_candidates_list = sentence['eventCandidates']
		for event in event_candidates_list:
			argumentslist = event['arguments']
			for argument in argumentslist:
				arg_index = argument['begin']
				token_index = event['begin'] 
				matrix_list.append( FV.get_feature_matrix_argument_prediction(token_index, arg_index, sentence, clf) )
				gold_list.append( argument['gold'] )

	if len(matrix_list) == 0:
		return None, None

	if clf=='perc':
		return matrix_list, gold_list
	elif clf=='nb':
		return vstack(matrix_list), gold_list


def build_dataset(file_list, FV, ind, kind='train', mode='trig', clf='nb', load=True):
	"""
	This function construct the data matrix X and target vector y.

	Arguments:
	- clf: string -> 'nb' for naivebayes, 'perc' for perceptron
	- kind: string -> 'train' or 'valid' or 'test'

	Output:
	- X: data matrix which depends per type of classifier specified by mode
	- y: vector of classes
	"""
	if load:
		print 'Loading X matrix and y from file.'
		file_name_pickle = "Xy_{0}_{1}_{2}_{3}.data".format(kind,mode,clf,ind)
		f = open(file_name_pickle,"rb")
		X, y = cPickle.load(f)
		f.close()
		return X, y
	else:
		if clf == 'nb':
			for file_index, file_name in enumerate(file_list):
				print 'Building test data from json file ',file_index , 'of', len(file_list)
				if mode == 'trig':
					if file_index == 0:
						X, y = build_trigger_data_batch(file_name, FV, clf='nb')
					else:
						(new_features, new_gold) = build_trigger_data_batch(file_name, FV, clf='nb')
						if new_features == None:
							continue
						else:
							X = vstack((X,new_features))
							y += new_gold
				if mode == 'arg':
					if file_index == 0:
						X, y = build_argument_data_batch(file_name, FV, clf='nb')
					else:
						(new_features, new_gold) = build_argument_data_batch(file_name, FV, clf='nb')
						if new_features == None:
							continue
						else:
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

		file_name_pickle = "Xy_{0}_{1}_{2}_{3}.data".format(kind,mode,clf,ind)
		f = open(file_name_pickle,"w")
		cPickle.dump((X,y),f)
		f.close()
		return X, y

def crossvalidation(file_list, k=5, mode='trig', clf='nb'):
	if mode=='trig':
		FV = feature_vector.FeatureVector('trigger')
	elif mode=='arg':
		FV = feature_vector.FeatureVector('argument')

	random.shuffle(file_list)
	chunks = np.array_split(file_list,k)
	chunks = np.asarray(chunks).tolist()

	for chunk in chunks:
		ind = chunks.index(chunk)
		train_list = chunks[:ind] + chunks[ind+1:]
		valid_list = chunk
		X_train, y_train = build_dataset(train_list, FV, ind=ind, kind='train', mode=mode, clf=clf, load=False)
		X_valid, y_valid = build_dataset(valid_list, FV, ind=ind, kind='valid', mode=mode, clf=clf, load=False)

		if clf=='nb':
			NB = nb.NaiveBayes()
			NB.train(np.asarray(X_train.todense()),np.asarray(y_train))
		elif clf=='perc':
			pass
	pass



#subsample the >None< events, to obtain more balanced data set.
def subsample(feature_list, trigger_list, clf, subsampling_rate = 0.75):
	"""
	clf: string -> 'perc' or 'nb'
	"""

	None_indices = [i for (i,trigger) in enumerate(trigger_list) if trigger == u'None']
	All_other_indices = [i for (i,trigger) in enumerate(trigger_list) if trigger != u'None']


	N = len(None_indices)
	N_pick = np.floor((1.0 - subsampling_rate) * N)
	#N_pick = len(All_other_indices)

	#now pick N_pick random 'None' samples among all of them.
	random_indices = np.floor(np.random.uniform(0, N , N_pick) )    
	subsample_of_None_indices = [None_indices[int(i)] for i in random_indices]

	# Identify indices of remaining samples after subsampling + randomise them.
	remaining_entries = subsample_of_None_indices + All_other_indices
	perm = np.random.permutation(len(remaining_entries))
	remaining_entries = [remaining_entries[p] for p in perm]

	# Return the subsampled list of samples.
	if clf=='perc':
		subsampled_feature_list = [feature_list[i] for i in remaining_entries ]
		subsampled_trigger_list = [trigger_list[i] for i in remaining_entries ]
		return subsampled_feature_list, subsampled_trigger_list
	elif clf=='nb':
		subsampled_feature_list = feature_list.tocsr()[remaining_entries].tocoo()
		subsampled_trigger_list = np.asarray([trigger_list[i] for i in remaining_entries ])
		return subsampled_feature_list, subsampled_trigger_list
    


def main():

	################### EXPLORATORY DATA ANALYSIS #############################

	# Just testing my functions a bit
	list_of_files = utils.list_files()

	x = utils.load_json_file(list_of_files[0])
	    
	# Finding and counting all event triggers
	t = utils.get_all_triggers(list_of_files)
	print("Number of distinct event triggers: {0}".format(len(t.keys())))
	pprint(t)

	# Finding and counting all possible arguments (=relationship labels)
	arg = utils.get_all_arguments(list_of_files)
	print("Number of relation arguments: {0}".format(len(arg.keys())))
	pprint(arg)

	########################## NAIVE BAYES ####################################

	## Naive Bayes on trigger

	# Read data
	print "Experiment 1: Naive Bayes predicting triggers"
	FV_trig = feature_vector.FeatureVector('trigger')
	train_list, valid_list = utils.create_training_and_validation_file_lists(list_of_files)

	X_train, y_train = build_dataset(train_list, FV_trig, ind=1, kind='train', mode='trig', clf='nb', load=True)
	X_train, y_train = subsample(X_train, y_train, clf='nb', subsampling_rate=0.60)
	X_valid, y_valid = build_dataset(valid_list, FV_trig, ind=1, kind='valid', mode='trig', clf='nb', load=True)

	NB_trig = nb.NaiveBayes()
	NB_trig.train(np.asarray(X_train.todense()),np.asarray(y_train))

	print "Evaluate Naive Bayes classifer predicting triggers on the train set..."
	CM, prec, rec, F1 = NB_trig.evaluate(np.asarray(X_train.todense()), np.asarray(y_train))
	print "Precision: {0}".format(prec)
	print "Recall: {0}".format(rec)
	print "F1-measure: {0}".format(F1)
	print "Confusion matrix:\n", np.int64(CM)

	print "Evaluate Naive Bayes classifer predicting triggers on the validation set..."
	CM, prec, rec, F1 = NB_trig.evaluate(np.asarray(X_valid.todense()), np.asarray(y_valid))
	print "Precision: {0}".format(prec)
	print "Recall: {0}".format(rec)
	print "F1-measure: {0}".format(F1)
	print "Confusion matrix:\n", np.int64(CM)

	## Naive Bayes on argument

	print "Experiment 2: Naive Bayes predicting arguments"
	FV_arg = feature_vector.FeatureVector('argument')

	X_train, y_train = build_dataset(train_list, FV_arg, ind=1, kind='train', mode='arg', clf='nb', load=True)
	X_train, y_train = subsample(X_train, y_train, clf='nb', subsampling_rate=0.90)
	X_valid, y_valid = build_dataset(valid_list, FV_arg, ind=1, kind='valid', mode='arg', clf='nb', load=True)

	NB_arg = nb.NaiveBayes()
	NB_arg.train(np.asarray(X_train.todense()), np.asarray(y_train))

	print "Evaluate Naive Bayes classifer predicting arguments on the train set..."
	CM, prec, rec, F1 = NB_arg.evaluate(np.asarray(X_train.todense()), np.asarray(y_train))
	print "Precision: {0}".format(prec)
	print "Recall: {0}".format(rec)
	print "F1-measure: {0}".format(F1)
	print "Confusion matrix:\n", np.int64(CM)

	print "Evaluate Naive Bayes classifer predicting arguments on the validation set..."
	CM, prec, rec, F1 = NB_arg.evaluate(np.asarray(X_valid.todense()), np.asarray(y_valid))
	print "Precision: {0}".format(prec)
	print "Recall: {0}".format(rec)
	print "F1-measure: {0}".format(F1)
	print "Confusion matrix:\n", np.int64(CM)

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
	# clf.fit(X_train,y_train)
	# print(clf.predict(X))


if __name__ == '__main__':
    main()
