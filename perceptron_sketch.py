# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 23:18:03 2014

@author: Johannes
"""

import numpy as np
import matplotlib.pyplot as plt
import feature_vector 
import utils
import json
import time
import warnings


#subsample the >None< events, to obtain more balanced data set.
def subsample(feature_list, trigger_list, subsampling_rate = 0.9):
    
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
    subsampled_feature_list = [feature_list[i] for i in remaining_entries ]
    subsampled_trigger_list = [trigger_list[i] for i in remaining_entries ]
    return subsampled_feature_list, subsampled_trigger_list



#generate one training batch in perceptron algorithm for event triggers. 
#output: For all events in file file_name: the features (matrix) & triggers
def build_trigger_data_batch(file_name, FV):
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
        matrix_list.append( FV.get_feature_matrix(token_index, sentence) )

    return matrix_list, trigger_list
            

#generate one training batch in perceptron algorithm for argument labels. 
#output: For all argument candidates in file file_name: 
#the features (matrix) & gold label of the trigger-argument relation
def build_argument_data_batch(file_name, FV):
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
                matrix_list.append( FV.get_feature_matrix_argument_prediction(token_index, 
                                                       arg_index, sentence) )
                gold_list.append( argument['gold'] )
    return matrix_list, gold_list
   
    
# create predictions for test set
def test_perceptron(FV, Lambda, file_list, mode, subsample = False):
    #Test data from all files in file_list
    feature_list = []
    gold_list = []
    for i_f, filename in enumerate(file_list):
        print 'Building test data from json file ',i_f , 'of', len(file_list)
        if mode == 'Trigger':
            (feat_list_one_file, gold_list_one_file) = build_trigger_data_batch(filename, FV)
        elif mode == 'Argument':
            (feat_list_one_file, gold_list_one_file) = build_argument_data_batch(filename, FV)
        else:
            warnings.warn('Error in test_perceptron: Must have mode "Trigger" or "Argument"!' )
            
        feature_list += feat_list_one_file
        gold_list += gold_list_one_file

    if subsample:
        print '###################################'
        print 'Nones before subsampling', gold_list.count(u'None'), 'of', len(gold_list)
        feature_list, gold_list = subsample(feature_list, gold_list, subsampling_rate = 0.95)    
        print 'Nones after subsampling', gold_list.count(u'None'), 'of',len(gold_list)






    predictions = []    
    gold_labels = []
    for i, (f,y) in enumerate(zip(feature_list, gold_list) ):
        if not i%100:
            print 'Predicting', i, 'of', len(gold_list)
        y_hat = predict(f, Lambda)
        predictions += [y_hat]
        if mode == 'Trigger':
            gold_labels += [ FV.trigger_list.index(y) ]
        elif mode == 'Argument':
            gold_labels += [ [u'None', u'Theme', u'Cause'].index(y) ]   
            
    return predictions, gold_labels
        
        
#predict function for perceptron
def predict(feature_matrix, Lambda, return_scores = False):
    #feature matrix: rows - classes; columns - feature dimensions
    scores = []
    for c in range(feature_matrix.shape[0]):
        scores.append( np.exp( feature_matrix.getrow(c).dot(Lambda[c,:])[0] ) )
        
    highest_score = max(scores)
    predicted_class = scores.index(highest_score)
    if return_scores:
        return scores
    else:
        return predicted_class


def train_perceptron(FV, training_files, T_max = 1, LR = 1.0, mode = 'Trigger'):
    t_start = time.time()
    N_files = len(training_files)
    
    #Generate training data
    feature_list = []
    gold_list = []
    for i_f, filename in enumerate(training_files):
        print 'Building training data from json file ',i_f

        if mode == 'Trigger':
            (feat_list_one_file, gold_list_one_file) = build_trigger_data_batch(filename, FV)
        elif mode == 'Argument':
            (feat_list_one_file, gold_list_one_file) = build_argument_data_batch(filename, FV)
        
        feature_list += feat_list_one_file
        gold_list += gold_list_one_file
    
    print '###################################'
    print 'Nones before subsampling', gold_list.count(u'None'), 'of', len(gold_list)
    if mode == 'Trigger':
        feature_list, gold_list = subsample(feature_list, gold_list, subsampling_rate = 0.6)    
    elif mode == 'Argument':
        feature_list, gold_list = subsample(feature_list, gold_list, subsampling_rate = 0.98)    
    print 'Nones after subsampling', gold_list.count(u'None'), 'of',len(gold_list)

    N_classes, N_dims = feature_list[0].shape
    N_samples = len(feature_list)

    #initialise parameters
    Lambda = np.random.normal(0.0, 1.0, [N_classes, N_dims])    
    iteration = 0
    misclassification_rates = []
    
    #start training epochs
    while iteration < T_max:
        iteration+=1
        misclassified = 0

        for sample in range(N_samples):
            X = feature_list[sample]
            gold = gold_list[sample]
            if mode == 'Trigger':
                y = FV.trigger_list.index(gold)
            elif mode == 'Argument':
                y = [u'None', u'Theme', u'Cause'].index(gold)
            
            y_hat = predict(X, Lambda)
            if not sample % 50:
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
    #Argument prediction
    FV_arg = feature_vector.FeatureVector('argument')
    train,valid = utils.create_training_and_validation_file_lists(ratio = 0.75, load=True)    
    Lambda, misclassification_rates = train_perceptron(FV_arg, train[:5], T_max = 10, LR = 10.0, mode='Argument')   
    plt.plot(misclassification_rates)

    (y_hat, y) = test_perceptron(FV_arg, Lambda, valid[:5], mode='Argument')
    errors = [1 for y1,y2 in zip(y_hat, y) if y1!=y2]
    validation_error = len(errors)/float(len(y))
    print (validation_error)
    utils.evaluate(y, y_hat, FV_arg, mode = 'Arguments')



if 1:
    #trigger prediction 
    FV_trig = feature_vector.FeatureVector('trigger')
    train,valid = utils.create_training_and_validation_file_lists(ratio = 0.75, load=True)    

    Lambda, misclassification_rates = train_perceptron(FV_trig, train[:100], T_max = 30, LR = 1.0, mode='Trigger')   
    plt.plot(misclassification_rates)

    (y_hat, y) = test_perceptron(FV_trig, Lambda, valid[:50], mode='Trigger')
    errors = [1 for y1,y2 in zip(y_hat, y) if y1!=y2]
    validation_error = len(errors)/float(len(y))  
    print (validation_error)
    utils.evaluate(y, y_hat, FV_trig, mode = 'Trigger')

    
    
if 1:
    plt.figure(2)
    plt.plot(np.transpose(Lambda))
    #plt.plot(np.transpose(Lambda)[-100:,:])
    #plt.xticks(range(len(symbols_list)), symbols_list, size='small')
    #plt.show()
    
    
    
    
    
    
    
    
    
    
