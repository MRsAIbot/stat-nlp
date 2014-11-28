# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 23:18:03 2014

@author: Johannes
"""

import numpy as np
import feature_vector as fv
import json


"""
Proposition for overall idea: (you can tear it apart completely if you want!)

a) Load and compute features for single sample on the run during the perceptron
iterations.
-->Not have to store the huge feature vectors for every single sample together.

b) alternative to a): generate a whole batch of samples at once
-->

-generate a single training datum using 'build_trigger_datum'

"""

def get_all_triggers(file_list):
	trigger_dict = defaultdict(int)
	for f in file_list:
		f_json = load_json_file(f)
		for i in range(len(f_json['sentences'])):
			trigger_list = f_json['sentences'][i]['eventCandidates']
			for trigger in trigger_list:
				trigger_dict[trigger['gold']] += 1

	return trigger_dict

#generate one training batch in perceptron algorithm for event triggers. 
#output: For all events in file file_name: the features (matrix) & triggers
def build_trigger_data_batch(file_name, FeatureVector):
    fv_gathered = []
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
    
    
    batch = FV.get_feature_batch(token_index_list, sentence_list, all_grammar_tags)
    #feature_vector = FV.get_vector_alternative(token_index, sentence, all_grammar_tags)
    #fv_gathered += [feature_vector]            
    return batch, triggers
                

def predict(feature_vector, Lambda, classes):
    class_probabilities = []
    for c in classes:
        class_probabilities.append( np.exp( feature_vector.dot(Lambda)[0] ) )
        
    highest_probability = max(class_probabilities)
    predicted_class = class_probabilities.index(highest_probability)
    return predicted_class
    
    


def run_perceptron(FeatureVector, t_list, lambda_init = None, T_max = 100):
    FV =feature_vector.FeatureVector(list_a)
    (batch, triggers) = build_trigger_data_batch(file_name, FV)
    
    N_samples, feature_dims = batch.shape

    if lambda_init == None:
        Lambda = np.random.normal(0.0, 1.0, feature_dims)
    else:
        Lambda = lambda_init
    
    iteration = 0
    while iteration < T_max:
        iteration+=1
        
        for r in range(N_samples):
            X = batch.getrow(r)
            y = triggers[r]
            
            y_hat = predict(X, Lambda, t_list)
            #can also use batch.dot(Lambda) to compute predictions for whole batch!!!
            if y_hat == y:
                pass
            else:
                Lambda = modify weights accordingly
        
    return Lambda
 
 

if 1:
    
    listOfFiles = list_files()
    file_name = listOfFiles[0]
    
    list_a = []
    list_a.append(feature_vector.phi_alternative_0)
    list_a.append(feature_vector.phi_alternative_0)
    list_a.append(feature_vector.phi_alternative_1)
    if 0:
        grammar_dict = feature_vector.identify_all_grammar_tags(listOfFiles)   
        all_grammar_tags = grammar_dict.keys()  #these lists should be saved and later loaded.

    FV =feature_vector.FeatureVector(list_a)
    (batch, triggers) = build_trigger_data_batch(file_name, FV)
    
    t_list = list(get_all_triggers(listOfFiles) )    
    
    
    
     if 0:
	listOfFiles = list_files()
	f1 = load_json_file(listOfFiles[0])
	sentence = f1['sentences'][0]   #pick first sentence
	token_index = 0 #first word in sentence

	
	f_v=feature_vector.FeatureVector(list_a)
	vec = f_v.get_vector_alternative( token_index, sentence, all_grammar_tags)
 
 
 
 
 