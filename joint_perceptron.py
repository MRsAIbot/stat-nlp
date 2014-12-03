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
import perceptron_sketch as perceptron


       
#predict function for perceptron
def predict_under_constraint(feature_matrix, Lambda, allowed_classes, 
                             return_scores = False):
    scores = []
    for c in allowed_classes:
        scores.append( np.exp( feature_matrix.getrow(c).dot(Lambda[c,:])[0] ) )
        
    highest_score = max(scores)
    predicted_class = scores.index(highest_score)
    if not return_scores:
        return predicted_class
    else:
        return predicted_class, scores


# which argument is cheapest to switch to 'Theme' [we need at least one!]?
def enforce_one_Theme(v, scores, Theme_argument):
    Theme_responses = [s[Theme_argument] for s in scores]
    if not u'Theme' in Theme_responses:
        Delta = []
        for j in range(len(v)):
            #check score difference between 'Theme' and unconstr. prediction
            Delta_j = scores[j][Theme_argument] - scores[j][v[j]]
            Delta += [Delta_j]
            
        convert = Delta.index(min(Delta))
        v[convert] = Theme_argument
    return v    #return old v if one argument was'Theme' already: no change.


#returns score of one particular combination of event and argument labels   
def total_score_joint(X_trigger, Lambda_trigger, X_arguments, Lambda_argument,
                event, arguments):
    
    ev_scores = perceptron.predict(X_trigger, Lambda_trigger,return_scores = True)
    s_e = ev_scores[event]
    
    #sum scores over all arguments
    s_a = 0.0
    for j in range(arguments):
        argument_scores = perceptron.predict(X_arguments, Lambda_argument, 
                                             return_scores = True)
        s_a += argument_scores[arguments[j]]

    return s_e + s_a



def argmax_joint_unconstrained(X_trig, Lambda_trig, X_arg, Lambda_arg, FV):
    
    None_trigger = FV.trigger_list.index(u'None')
    None_argument = FV.arguments_list.index(u'None')
    Cause_argument = FV.arguments_list.index(u'Cause')
    Theme_argument = FV.arguments_list.index(u'Theme')
    
    #COMPUTE s1: None trigger and None arguments always
    e_1 = None_trigger
    a_1 = [None_argument]*len(X_arg)
    s1 = total_score_joint(X_trig, Lambda_trig, X_arg, Lambda_arg, e_1, a_1)
                     
                     

    #COMPUTE s2: Regulation triggers allowed
    legal_classes = range(10)
    legal_classes.remove(None_trigger)
    e_2 = predict_under_constraint(X_trig, Lambda_trig, legal_classes)

    a_2 = []
    scores = []
    for j in range(len(Lambda_arg)):
        a_j, sco = predict_under_constraint(X_arg[j], Lambda_arg, range(3), 
                                                return_scores = True) 
        a_2  += [ a_j ]
        scores  += [ sco ]
        
    #check if Theme appears at least once. If not: adjust to second best score
    a_2 = enforce_one_Theme(a_2, scores, Theme_argument)   
    s2 = total_score_joint(X_trig, Lambda_trig, X_arg, Lambda_arg, e_2, a_2 )

    
    
    #COMPUTE s3: no regulation triggers or regulation events allowed
    legal_triggers = range(10)
    legal_triggers.remove(None_trigger)
    #identify the trigger indices for regulation events
    Regulation_triggers = [i for i in range(len(FV.trigger_list)) \
                           if 'egulation' in FV.trigger_list[i]]
     
    for r in Regulation_triggers:
        legal_triggers.remove(r)
    e_3 = predict_under_constraint(X_trig, Lambda_trig, legal_triggers)
    
    #argument & scores: Cause arguments not allowed
    legal_arglabels = range(3)
    legal_arglabels.remove(Cause_argument)
    
    a_3 = []
    scores = []
    for j in range(len(Lambda_arg)):
        a_j, sco = predict_under_constraint(X_arg[j], Lambda_arg, legal_arglabels, 
                                            return_scores = True) 
        a_3  += [ a_j ]
        scores  += [ sco ]
        
    a_3 = enforce_one_Theme(a_3, scores, Theme_argument)    
    s3 = total_score_joint(X_trig, Lambda_trig, X_arg, Lambda_arg, e_3, a_3 )
    
    
    #identify highest scoring case; return its parameters
    best_score = [s1,s2,s3].index(max([s1,s2,s3]))
    if best_score == 0:
        return (e_1, a_1)
    elif best_score == 1:
        return (e_2, a_2)
    else:
        return (e_3, a_3)
    
        


def predict_joint(X_trigger, Lambda_trigger, X_arguments, Lambda_argument, mode):
    """
    e: event trigger [10 possible event triggers]
    a: vector of argument labels. Each element is from [None, Theme, Cause]
    """
    if mode == 'unconstrained':
        #predictions can be computed individually, since joint prob factorises 
        e_hat = perceptron.predict(X_trigger, Lambda_trigger)
        a_hat = []
        for arg in range(len(X_arguments))  :
            a_hat += [perceptron.predict(X_arguments[arg], Lambda_argument)]
        return e_hat, a_hat
            
    elif mode == 'constrained':
        e_hat, a_hat = argmax_joint_unconstrained(X_trigger, Lambda_trigger, 
                                                  X_arguments, Lambda_argument, FV)
                                                  
    else:
        warnings.warn(' ### Problem in predict_joint: mode must be either ' \
                        + '"constrained" or "unconstrained". ###')
        
    
    
             
    
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
                                             X_arguments, Lambda_argument, mode)
                                             
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
                for j in range(len(y_arguments)):
                    Delta_arg = np.zeros([N_classes_argument, N_dims_argument])
                    Delta_arg[y_hat_a[j] , :] = -LR * X_arguments[j].getrow(y_hat_a[j]).todense() 
                    Delta_arg[y_arguments[j] , :] = LR * X_arguments[j].getrow(y_arguments[j]).todense() 
    
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