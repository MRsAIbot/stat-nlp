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



#generate one training batch in perceptron algorithm for event triggers
def build_trigger_data_batch(file_name):
    data = []
    f_json = json.load(open(file_name))
    for sentence in f_json['sentences']:
        event_candidates_list = sentence['eventCandidates']
        for event in event_candidates_list:
            feature_vector = fv.compute_features_trigger(sentence, event)
            trigger = event['gold']
            #only provisional, maybe better save in (sparse?) matrix form
            data += [(feature_vector, trigger)]
            
    return data
                

def predict(feature_vector, Lambda, classes):
    class_probabilities = []
    for c in classes:
        class_probabilities += np.exp( np.dot(feature_vector, Lambda) )
    highest_probability = max(class_probabilities)
    predicted_class = class_probabilities.index(highest_probability)
    return predicted_class
    

def run_perceptron(FeatureVector, lambda_init = None, T_max = 100):
    lambda_init = np.random.normal(0.0, 1.0, 100)
    
    iteration = 0
    while iteration < T_max:
        iteration+=1
        
        (X,Y) = build_trigger_data_batch(...)
        for (x,y) in zip(X,Y):
            y_hat = predict(feature_vector, Lambda, classes)
            if y_hat == y:
                pass
            else:
                Lambda = modify weights accordingly
        
    return Lambda
 
 
 
 
 
 
 
 
 