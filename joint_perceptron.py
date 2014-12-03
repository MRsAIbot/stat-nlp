# -*- coding: utf-8 -*-
"""
Created on Wed Dec 03 16:40:50 2014

@author: Johannes
"""

import utils
import feature_vector
import warnings
import json
import time
import numpy as np


def argmax_joint_unconstrained():
    pass





    
    
    
def argmax_joint_constrained():
    pass



def predict_joint(X_trigger, Lambda_trigger, X_arguments, Lambda_argument):
    
    pass
    
    
    
             
    
def build_joint_data_batch(file_name, FV):
    gold_list = []
    feature_matrix_list = []   #list of feature matrices for both trigger & argument
    f_json = json.load(open(file_name)) 
    
    for sentence in f_json['sentences']:
        event_candidates_list = sentence['eventCandidates']
        
        for event in event_candidates_list:
            argumentslist = event['arguments']
            gold_trigger = event['gold']
            token_index = event['begin']
            trigger_matrix = FV.get_feature_matrix(token_index, sentence) 
            
            gold_arguments = []
            argument_matrices = []
            for argument in argumentslist:
                arg_index = argument['begin']
                gold_arguments.append( argument['gold'] )
                argument_matrices.append( FV.get_feature_matrix_argument_prediction(token_index, 
                                                       arg_index, sentence) )
    
            gold_list += [(gold_trigger, gold_arguments)]
            feature_matrix_list += [(trigger_matrix, argument_matrices)]
            
    return feature_matrix_list, gold_list
    
    
    
            






def train_perceptron_joint(FV, training_files, T_max = 1, LR = 1.0, mode = 'unconstrained'):
    if mode == 'Joint_unconstrained' or mode == 'Joint_constrained':
        pass
    else: 
        warnings.warn('Error in train_perceptron_joint: Must have mode ' \
        + '"Joint_unconstrained" or"Joint_constrained" ' ) 

    t_start = time.time()
    N_files = len(training_files)
    
    #Generate training data
    feature_list = []
    gold_list = []
    for i_f, filename in enumerate(training_files):
        print 'Building training data from json file ',i_f
        (feat_list_one_file, gold_list_one_file) = build_joint_data_batch(filename, FV)
        
        feature_list += feat_list_one_file
        gold_list += gold_list_one_file
    
    #feature_list, gold_list = perceptron.subsample(feature_list, gold_list, subsampling_rate = 0.95)    
    N_classes_trigger, N_dims_trigger = feature_list[0][0].shape
    N_classes_argument, N_dims_argument = feature_list[0][1][0].shape
    
    N_samples = len(feature_list)

    #initialise parameters
    Lambda_trigger = np.random.normal(0.0, 1.0, [N_classes_trigger, N_dims_trigger])    
    Lambda_argument = np.random.normal(0.0, 1.0, [N_classes_argument, N_dims_argument])    
    
    iteration = 0
    misclassification_rates = []
    
    #start training epochs
    while iteration < T_max:
        iteration+=1
        misclassified = 0

        for sample in range(N_samples):
            X_trigger = feature_list[sample][0]
            X_arguments = feature_list[sample][1]
            
            gold_trigger = gold_list[sample][0]
            gold_arguments = gold_list[sample][1]
            
            y_trigger = FV.trigger_list.index(gold_trigger)
            y_arguments = [ [u'None', u'Theme', u'Cause'].index(ga) for ga in gold_arguments]
            
            y_hat_e, y_hat_a = predict_joint(X_trigger, Lambda_trigger, 
                                             X_arguments, Lambda_argument)
                                             
            if not sample % 50:
                print 'it',iteration, sample, 'of', N_samples,'Predict:', y_trigger, 'True:', y_hat_e
                
            if y_hat_e != y_trigger or y_hat_a != y_arguments:
                misclassified +=1
                #adjust trigger weights
                Delta_trigger = np.zeros([N_classes_trigger, N_dims_trigger])
                Delta_trigger[y_hat_e, :] = -LR * X_trigger.getrow(y_hat_e).todense() 
                Delta_trigger[y_trigger , :] = LR * X_trigger.getrow(y_trigger).todense() 

                Lambda_New = np.add(Lambda_trigger, Delta_trigger)
                Lambda_trigger = Lambda_New
                
                #adjust argument weights
                for arg in range(y_arguments):
                    Delta_arg = np.zeros([N_classes_argument, N_dims_argument])
                    Delta_arg[y_hat_a[arg] , :] = -LR * X_arguments.getrow(y_hat_a[arg]).todense() 
                    Delta_arg[y_arguments[arg] , :] = LR * X_arguments.getrow(y_arguments[arg]).todense() 
    
                    Lambda_New = np.add(Lambda_argument, Delta_arg)
                    Lambda_argument = Lambda_New
            else:
                pass #prediction correct, no change.
                
        print misclassified
        misclassification_rates += [ float(misclassified)/float(N_samples) ]
        
    print time.time()-t_start, 'sec for', N_files, 'Files and', T_max, 'epochs.'
    return Lambda_trigger, Lambda_argument, misclassification_rates
 





            
            
            
            
if 1:
    #joint prediction
    FV_joint = feature_vector.FeatureVector('joint')
    FV = FV_joint
    train,valid = utils.create_training_and_validation_file_lists(ratio = 0.75, load=True)    

    L_t, L_a, misc = train_perceptron_joint(FV, train[0:1], T_max = 1, 
                                            LR = 1.0, mode = 'unconstrained')