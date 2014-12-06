# -*- coding: utf-8 -*-
"""
Created on Sat Dec 06 15:39:29 2014

@author: Johannes
"""

import utils
import os
import joint_perceptron as jp
import feature_vector
import cPickle
import json

test_path_inputs ='C:/Python27/aaa_UCL/Natural Language Processing/assignment2/bionlp2011genia-statnlp-test-clean/*.json'
test_files_output_dir = 'C:/Python27/aaa_UCL/Natural Language Processing/assignment2/predictions/'

test_files_list = utils.list_files(path=test_path_inputs)
if not os.path.exists(test_files_output_dir):
    os.makedirs(test_files_output_dir)
    
    

FV_joint = feature_vector.FeatureVector('joint')
FV = FV_joint
with open('Joint_perceptron.data', 'rb') as f:
    Lambda_e, Lambda_a = cPickle.load(f)
    
    
#must have L_t,L_a in working memory!
    
evaluate_test_list = test_files_list[:3]


    
    
test_file = test_files_list[1]
for i_f,test_file in enumerate(evaluate_test_list):
    print 'Test File', i_f, 'of' , len(evaluate_test_list)
    
    #generate predictions for current file
    (p_e,g_e, p_a, g_a) = jp.test_perceptron_joint(FV, Lambda_e, Lambda_a, 
                        [test_file], mode = 'Joint_unconstrained', 
                        subsample = False)
                        
    f_fill_this = utils.load_json_file(test_file)    
    counter_e = 0
    counter_a = 0
    for sentence in f_fill_this['sentences']:
        event_candidates = sentence['eventCandidates']
        for ec in event_candidates:
            ec['predicted'] = FV.trigger_list[p_e[counter_e]]
            counter_e +=1
    if counter_e != len(p_e):
        print 'PROBLEM: LENGTH OF PREDICTION VECTOR (trigger) DOESNT FIT!'
        
    #save resulting dictionary to output file
    output_file_name = test_file.split('\\')[-1]
    output_path = test_files_output_dir + output_file_name
    with open(output_path, 'wb') as f_out:
        json.dump(f_fill_this, f_out)
        
 

if 0:    
    filename_in='C:/Python27/aaa_UCL/Natural Language Processing/assignment2/predictions/PMC-1134658-00-TIAB.json'
    with open(filename_in, 'rb') as f_in:
        f_reloaded2 = json.load(f_in)
            
            
