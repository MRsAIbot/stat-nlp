# -*- coding: utf-8 -*-
"""
Created on Wed Dec 03 16:40:50 2014

@author: Johannes
"""

import utils
import feature_vector
import perceptron_sketch as perceptron

import warnings
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import cPickle




#first subsample a set of event candidates with lower rate of 'None' triggers
#then also subsample the argument candidates.
def subsample_jointly(feature_tuples, gold_tuples, rate_trig, rate_arg):
    
    None_trig_indices = [i for (i,trigger) in enumerate(gold_tuples) if trigger[0] == u'None']

    All_other_trig_indices = range(len(gold_tuples))
    for n_i in None_trig_indices:
        All_other_trig_indices.remove(n_i)

    
    N = len(None_trig_indices)
    N_pick = np.floor((1.0 - rate_trig) * N)

    random_indices_trig = np.int16(np.floor(np.random.uniform(0, N , N_pick) ) )
    subsample_trig_indices = [None_trig_indices[i] for i in random_indices_trig \
                              if len(gold_tuples[i][1]) > 0]

    # Identify indices of remaining samples after subsampling + randomise them.
    remaining_entries = subsample_trig_indices + All_other_trig_indices
    perm = np.random.permutation(len(remaining_entries))
    remaining_entries = [remaining_entries[p] for p in perm]

    subsampled_feature_tuples = [feature_tuples[i] for i in remaining_entries ]
    subsampled_gold_tuples = [gold_tuples[i] for i in remaining_entries ]


    #Now subsample the argument candidates within each remaining event candidate
    final_f_tuples = []
    final_g_tuples = []
    
    for (f_tuple, g_tuple) in zip(subsampled_feature_tuples,subsampled_gold_tuples):
        m = len(g_tuple[1])
        None_arg_indices = []
        All_other_indices = range(m)
        for j in range(m):
            if g_tuple[1][j] == u'None':
                All_other_indices.remove(j)
                None_arg_indices += [j]
        
                
        N = len(None_arg_indices)
        N_pick = np.floor((1.0 - rate_arg) * N)
        if N != 0:
            N_pick = max(N_pick, 1)
        
        random_indices_arg = np.floor(np.random.uniform(0, N , N_pick) )    
        subsample_arg_indices = [None_arg_indices[int(i)] for i in random_indices_arg]
    
        # Identify indices of remaining arguments after subsampling and pick.
        remaining_entries = subsample_arg_indices + All_other_indices

        f_arg_subsampled = [f_tuple[1][i] for i in remaining_entries ]
        g_arg_subsampled = [g_tuple[1][i] for i in remaining_entries ]
    
        final_f_tuples += [(f_tuple[0], f_arg_subsampled) ]
        final_g_tuples += [(g_tuple[0], g_arg_subsampled) ]

    return final_f_tuples, final_g_tuples












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
            #Delta_j = scores[j][Theme_argument] - scores[j][v[j]]
            Delta_j = scores[j][v[j]] - scores[j][Theme_argument] 
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
    for j in range(len(arguments)):
        argument_scores = perceptron.predict(X_arguments[j], Lambda_argument, 
                                             return_scores = True)
        s_a += argument_scores[arguments[j]]

    return s_e + s_a



def argmax_joint_unconstrained(X_trig, Lambda_trig, X_arg, Lambda_arg, FV):
    
    None_trigger = FV.trigger_list.index(u'None')
    None_argument = FV.arguments_list.index(u'None')
    Cause_argument = FV.arguments_list.index(u'Cause')
    Theme_argument = FV.arguments_list.index(u'Theme')
    
    #COMPUTE s1: case trigger==None and all arguments are None
    e_1 = None_trigger
    a_1 = [None_argument]*len(X_arg)
    s1 = total_score_joint(X_trig, Lambda_trig, X_arg, Lambda_arg, e_1, a_1)
                     
                     

    #COMPUTE s2: ONLY Regulation triggers are allowed
    #identify the trigger indices for regulation events [also pos/neg-regul.]
    Regulation_triggers = [i for i in range(len(FV.trigger_list)) \
                           if 'egulation' in FV.trigger_list[i]]
    legal_triggers = Regulation_triggers

    e_2 = predict_under_constraint(X_trig, Lambda_trig, legal_triggers)
    a_2 = []
    scores = []
    for j in range(len(X_arg)):
        #generate free predictions, might so far miss a Theme argument 
        a_j, sco = predict_under_constraint(X_arg[j], Lambda_arg, range(3), 
                                                return_scores = True) 
        a_2  += [ a_j ]
        scores  += [ sco ]
        
    #check if Theme appears at least once. If not: adjust to second best score
    a_2 = enforce_one_Theme(a_2, scores, Theme_argument)   
    s2 = total_score_joint(X_trig, Lambda_trig, X_arg, Lambda_arg, e_2, a_2 )

    
    
    #COMPUTE s3: no regulation [also pos/neg regul.] triggers or None trigger
    #allowed. Restrict possible argument labels to None and Theme [skip Cause].
    legal_triggers = range(10)
    for r in Regulation_triggers:
        legal_triggers.remove(r)
    legal_triggers.remove(None_trigger)
    e_3 = predict_under_constraint(X_trig, Lambda_trig, legal_triggers)
    
    #argument & scores: Cause arguments not allowed
    legal_arglabels = range(3)
    legal_arglabels.remove(Cause_argument)
    
    a_3 = []
    scores = []
    for j in range(len(X_arg)):
        a_j, sco = predict_under_constraint(X_arg[j], Lambda_arg, legal_arglabels, 
                                            return_scores = True) 
        a_3  += [ a_j ]
        scores  += [ sco ]
    a_3 = enforce_one_Theme(a_3, scores, Theme_argument)    
    s3 = total_score_joint(X_trig, Lambda_trig, X_arg, Lambda_arg, e_3, a_3 )
    
    
    #Compare s1,s2,s3. Identify highest scoring case; return its parameters
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
    if mode == 'Joint_unconstrained':
        #predictions can be computed individually, since joint prob factorises 
        e_hat = perceptron.predict(X_trigger, Lambda_trigger)
        a_hat = []
        for j in range(len(X_arguments))  :
            a_hat += [perceptron.predict(X_arguments[j], Lambda_argument)]
        return e_hat, a_hat
            
    elif mode == 'Joint_constrained':
        e_hat, a_hat = argmax_joint_unconstrained(X_trigger, Lambda_trigger, 
                                                  X_arguments, Lambda_argument, FV)
        return e_hat, a_hat
                                                  
    else:
        warnings.warn(' ### Problem in predict_joint: mode must be either ' \
                        + '"constrained" or "unconstrained". ###')
        
    
    
             
    
def build_joint_data_batch(file_name, FV, subsample = True):
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
    if subsample:
        return subsample_jointly(feature_matrix_list, gold_list, rate_trig=.9, rate_arg=.9)
    else: 
        return (feature_matrix_list, gold_list)


# create predictions for test set
def test_perceptron_joint(FV, Lambda_trig, Lambda_arg, file_list, mode, subsample = False):
    #Test data from all files in file_list
    feature_list = []
    gold_list = []
    for i_f, file_name in enumerate(file_list):
        print 'Building test data from json file ',i_f , 'of', len(file_list)
        (feat_list_one_file, gold_list_one_file) = build_joint_data_batch(file_name, FV, subsample)
        feature_list += feat_list_one_file
        gold_list += gold_list_one_file

    if subsample:
        print '###################################'
        print 'Nones before subsampling', gold_list.count(u'None'), 'of', len(gold_list)
        feature_list, gold_list = subsample(feature_list, gold_list, subsampling_rate = 0.95)    
        print 'Nones after subsampling', gold_list.count(u'None'), 'of',len(gold_list)

    predictions_e = []    
    predictions_a = []
    gold_e = []
    gold_a = []
    N_samples = len(feature_list)

    for sample in range(N_samples):
        X_trigger = feature_list[sample][0]
        X_arguments = feature_list[sample][1]
        
        gold_trigger = gold_list[sample][0]
        gold_arguments = gold_list[sample][1]
        
        gold_e += [FV.trigger_list.index(gold_trigger)]
        gold_a += [ [ FV.arguments_list.index(ga) for ga in gold_arguments] ]
        
        y_hat_e, y_hat_a = predict_joint(X_trigger, Lambda_trig, \
                                         X_arguments, Lambda_arg, mode)    
        predictions_e += [y_hat_e]    
        predictions_a += [y_hat_a]
    
    print len(predictions_e)            
    return predictions_e, gold_e, predictions_a, gold_a



def train_perceptron_joint(FV, training_files, T_max = 1, LR = 1.0, 
                           mode = 'Joint_unconstrained', plot = True):
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
    misclassification_rates_t = []
    misclassification_rates_a = []
    
    #start training epochs
    while iteration < T_max:
        iteration+=1
        misclassified_t = 0
        misclassified_a = 0

        for sample in range(N_samples):
            X_trigger = feature_list[sample][0]
            X_arguments = feature_list[sample][1]
            
            gold_trigger = gold_list[sample][0]
            gold_arguments = gold_list[sample][1]
            
            y_trigger = FV.trigger_list.index(gold_trigger)
            y_arguments = [ FV.arguments_list.index(ga) for ga in gold_arguments]
            
            y_hat_e, y_hat_a = predict_joint(X_trigger, Lambda_trigger, \
                                             X_arguments, Lambda_argument, mode)
                                             
            if not sample % 50:
                print 'it',iteration, sample, 'of', N_samples,'Predict:', y_hat_e, 'True:', y_trigger
                
            if y_hat_e != y_trigger:
                misclassified_t +=1
                #adjust trigger weights
                Delta_trigger = np.zeros([N_classes_trigger, N_dims_trigger])
                Delta_trigger[y_hat_e, :] = -LR * X_trigger.getrow(y_hat_e).todense() 
                Delta_trigger[y_trigger , :] = LR * X_trigger.getrow(y_trigger).todense() 

                Lambda_New = np.add(Lambda_trigger, Delta_trigger)
                Lambda_trigger = Lambda_New
            if y_hat_a != y_arguments:
                misclassified_a +=1
                #adjust argument weights
                for j in range(len(y_arguments)):
                    Delta_arg = np.zeros([N_classes_argument, N_dims_argument])
                    Delta_arg[y_hat_a[j] , :] = -LR * X_arguments[j].getrow(y_hat_a[j]).todense() 
                    Delta_arg[y_arguments[j] , :] = LR * X_arguments[j].getrow(y_arguments[j]).todense() 
    
                    Lambda_New = np.add(Lambda_argument, Delta_arg)
                    Lambda_argument = Lambda_New
            else:
                pass #prediction correct, no change.predict_joint
                
        misclassification_rates_t += [ float(misclassified_t)/float(N_samples) ]
        misclassification_rates_a += [ float(misclassified_a)/float(N_samples) ]
        
    print time.time()-t_start, 'sec for', N_files, 'Files and', T_max, 'epochs.'
    if plot:
        plt.plot(misclassification_rates_t, 'b', label='trigger')
        plt.plot(misclassification_rates_a, 'g', label='argument')
        plt.legend()
    return Lambda_trigger, Lambda_argument, misclassification_rates_t, misclassification_rates_a
 





#BEFORE RUNNING: make all if 0: in perceptron_sketch because it is imported.
            
            
            
if 0:
    #joint prediction
    FV_joint = feature_vector.FeatureVector('joint')
    FV = FV_joint
    train,valid = utils.create_training_and_validation_file_lists(ratio = 0.75, load=True)    

    L_t, L_a, misc_t_,misc_a = train_perceptron_joint(FV, train[:10], T_max = 50, 
                                            LR = 10.0, mode = 'Joint_unconstrained')

    (p_e,g_e, p_a, g_a) = test_perceptron_joint(FV, L_t, L_a, valid[:5], 
                                                mode = 'Joint_unconstrained', 
                                                subsample = False)
    
    
    savedata = (L_t, L_a)
    with open('Joint_perceptron.data', 'wb') as f:
        cPickle.dump(savedata, f)  
    with open('Joint_perceptron.data', 'rb') as f:
        LL_t, LL_a = cPickle.load(f)
    
    #evaluate trigger predictions
    errors_e = [1 for y1,y2 in zip(p_e, g_e) if y1!=y2]
    validation_error_e = len(errors_e)/float(len(g_e))  
    print 'Trigger prediction error (accuracy)',(validation_error_e)
    utils.evaluate(p_e, g_e, FV, mode = 'Trigger')
    
    #evaluate argument predictions
    pp_a = [i for sublist in p_a for i in sublist]
    gg_a = [i for sublist in g_a for i in sublist]
    errors_a = [1 for y1,y2 in zip(pp_a, gg_a) if y1!=y2]
    validation_error_a = len(errors_a)/float(len(gg_a))  
    print 'Argument prediction error (accuracy)',(validation_error_a)
    utils.evaluate(pp_a, gg_a, FV, mode = 'Arguments')
    

    
if 0:
    plt.figure(2)
    plt.subplot(211)
    plt.plot(np.transpose(L_t))
    plt.subplot(212)
    plt.plot(np.transpose(L_a))