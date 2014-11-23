# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 23:18:03 2014

@author: Johannes
"""

import numpy as np
import assignment2 as ass2



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
def build_trigger_data_batch(filename):
    data = []
    f_json = ass2.load_json_file(filename)
    for sentence in f_json['sentences']:
        event_candidates_list = sentence['eventCandidates']
        for event in event_candidates_list:
            feature_vector = compute_features_trigger(sentence, event)
            trigger = event['gold']
            #only provisional, maybe better save in (sparse?) matrix form
            data += [(feature_vector, trigger)]
            
    return data
                
    
#compute feature vector for given event (=dict) in its sentence (also dict)
def compute_features_trigger(sentence, event):
    pass        
            
            
#for running perceptron algorithm to detect arguments (=labels of relations)
def build_trigger_data_batch(filename):
    pass

