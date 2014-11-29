# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 23:18:03 2014

@author: Johannes
"""

import numpy as np
import feature_vector 
import utils
import json
import time
from collections import defaultdict

"""
Proposition for overall idea: (you can tear it apart completely if you want!)

a) Load and compute features for single sample on the run during the perceptron
iterations.
-->Not have to store the huge feature vectors for every single sample together.

b) alternative to a): generate a whole batch of samples at once
-->

-generate a single training datum using 'build_trigger_datum'

"""


def subsample(feature_list, trigger_list, subsampling_rate = 0.95):
    
    None_indices = [i for (i,trigger) in enumerate(trigger_list) if trigger == u'None']
    All_other_indices = [i for (i,trigger) in enumerate(trigger_list) if trigger != u'None']
    
    N = len(None_indices)
    N_pick = np.floor((1.0 - subsampling_rate) * N)
    
    #now pick N_pick random 'None' samples among all of them.
    random_indices = np.floor(np.random.uniform(0, N , N_pick) )    
    subsample_of_None_indices = [None_indices[int(i)] for i in random_indices]
    
    # Identify indices of remaining samples after subsampling + randomise them.
    remaining_entries = subsample_of_None_indices + All_other_indices
    perm = np.random.permutation(len(remaining_entries))
    remaining_entries = [remaining_entries[p] for p in perm]
    
    # Return the subsampled list of samples.
    subsampled_feature_list = [feature_list[i] for i in remaining_entries ]
    subsampled_trigger_list = [trigger_list[i] for i in remaining_entries ]
    return subsampled_feature_list, subsampled_trigger_list


#generate one training batch in perceptron algorithm for event triggers. 
#output: For all events in file file_name: the features (matrix) & triggers
def build_trigger_data_batch(file_name, FV):
    triggers = []
    
    token_index_list = []
    sentence_list = []
    f_json = json.load(open(file_name))
    
    for sentence in f_json['sentences']:
        event_candidates_list = sentence['eventCandidates']
        for event in event_candidates_list:
            token_index_list.append( event['begin'] )
            sentence_list.append(sentence)
            triggers += [ event['gold'] ]
    
    matrix_list = []
    for token_index,sentence in zip(token_index_list,sentence_list):
        matrix_list.append( FV.get_feature_matrix(token_index, sentence, all_grammar_tags) )
        
    #batch = FV.get_feature_batch(token_index_list, sentence_list, all_grammar_tags)
    #feature_vector = FV.get_vector_alternative(token_index, sentence, all_grammar_tags)
    return matrix_list, triggers
            
            
def test_perceptron(Lambda):
    pass        


def predict(feature_matrix, Lambda, N_classes):
    #feature matrix: rows - classes; columns - feature dimensions
    #FV.get_feature_matrix(token_index, sentence, all_grammar_tags)
    scores = []
    for c in range(N_classes):
        scores.append( np.exp( feature_matrix.getrow(c).dot(Lambda[c,:])[0] ) )
        
    highest_score = max(scores)
    predicted_class = scores.index(highest_score)
    return predicted_class
    
    


def train_perceptron(FV, t_list, N_files, T_max = 1, LR = 1.0):
    #FV =feature_vector.FeatureVector(list_a)
    #(batch, triggers) = build_trigger_data_batch(file_name, FV)

    #f0 = FV.get_feature_matrix(token_index, sentence, all_grammar_tags)
    t_start = time.time()

    #Generate training data
    feature_list = []
    trigger_list = []
    for i_f, filename in enumerate(listOfFiles[0:N_files]):
        print 'Building training data from json file ',i_f
        (feat_list_one_file, trig_list_one_file) = build_trigger_data_batch(listOfFiles[0], FV)
        feature_list += feat_list_one_file
        trigger_list += trig_list_one_file
        

    feature_list, trigger_list = subsample(feature_list, trigger_list, subsampling_rate = 0.95)    
        
    
    N_classes, N_dims = feature_list[0].shape
    N_samples = len(trigger_list)


    Lambda = np.random.normal(0.0, 1.0, [N_classes, N_dims])

    
    iteration = 0
    misclassification_rates = []
    
    #start training epochs
    while iteration < T_max:
        iteration+=1
        misclassified = 0

        for sample in range(N_samples):
            X = feature_list[sample]
            trigger = trigger_list[sample] 
            y = t_list.index(trigger)
            
            y_hat = predict(X, Lambda, N_classes)
            print 'it',iteration, sample, 'of', N_samples,'Predict:', y, 'True:', y_hat
            if y_hat != y:
                Delta = np.zeros([N_classes, N_dims])
                Delta[y_hat, :] = -LR*X.getrow(y_hat).todense() 
                Delta[y , :]  =  LR*X.getrow(y).todense() 

                Lambda_New = np.add(Lambda, Delta)
                Lambda = Lambda_New
                
                misclassified +=1
            else:
                pass #prediction correct, no change.
        print misclassified
        misclassification_rates += [ float(misclassified)/float(N_samples) ]
        
    print time.time()-t_start, 'sec for', N_files, 'Files and', T_max, 'epochs.'
    return Lambda, misclassification_rates
 
 

if 0:
    
    listOfFiles = utils.list_files()
    file_name = listOfFiles[0]
    triggers = list(utils.get_all_triggers(listOfFiles) )
    
    list_a = []
    list_a.append(feature_vector.phi_alternative_0)
    list_a.append(feature_vector.phi_alternative_0)
    list_a.append(feature_vector.phi_alternative_1)
    if 0:
        grammar_dict = feature_vector.identify_all_grammar_tags(listOfFiles)   
        all_grammar_tags = grammar_dict.keys()  #these lists should be saved and later loaded.

    FV = feature_vector.FeatureVector(list_a)
    (batch, triggers) = build_trigger_data_batch(file_name, FV)
    
    t_list = list(utils.get_all_triggers(listOfFiles) )    
    
    if 1:
        f1 = utils.load_json_file(listOfFiles[0])
        sentence = f1['sentences'][0]   #pick first sentence
        token_index = 0 #first word in sentence
        feature_matrix = FV.get_feature_matrix(token_index, sentence, all_grammar_tags)
    
    
    
    Lambda, misclassification_rates = train_perceptron(FV, t_list, 50, T_max = 5, LR = 1.0)   
    
    
    train,valid = utils.create_training_and_validation_file_lists(ratio = 0.75)    
    
    
    
    """
     if 0:
        	

	
        	f_v=feature_vector.FeatureVector(list_a)
        	vec = f_v.get_vector_alternative( token_index, sentence, all_grammar_tags)
         
    """
 
 
 