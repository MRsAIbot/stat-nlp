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


#subsample the >None< events, to obtain more balanced data set.
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
            
            
            
def test_perceptron(FV, Lambda, file_list):
    #Test data from all files in file_list
    feature_list = []
    trigger_list = []
    for i_f, filename in enumerate(file_list):
        print 'Building test data from json file ',i_f , 'of', len(file_list)
        (feat_list_one_file, trig_list_one_file) = build_trigger_data_batch(filename, FV)
        feature_list += feat_list_one_file
        trigger_list += trig_list_one_file

    predictions = []    
    gold_labels = []
    for i, (f,y) in enumerate(zip(feature_list, trigger_list) ):
        if not i%100:
            print 'Predicting', i, 'of', len(trigger_list)
        y_hat = predict(f, Lambda)
        predictions += [y_hat]
        gold_labels += [ FV.trigger_list.index(y) ]
        
    return predictions, gold_labels
        
        

def predict(feature_matrix, Lambda):
    #feature matrix: rows - classes; columns - feature dimensions
    scores = []
    for c in range(feature_matrix.shape[0]):
        scores.append( np.exp( feature_matrix.getrow(c).dot(Lambda[c,:])[0] ) )
        
    highest_score = max(scores)
    predicted_class = scores.index(highest_score)
    return predicted_class
    
    


def train_perceptron(FV, training_files, T_max = 1, LR = 1.0):
    t_start = time.time()

    #Generate training data
    N_files = len(training_files)
    feature_list = []
    trigger_list = []
    for i_f, filename in enumerate(training_files):
        print 'Building training data from json file ',i_f
        (feat_list_one_file, trig_list_one_file) = build_trigger_data_batch(filename, FV)
        feature_list += feat_list_one_file
        trigger_list += trig_list_one_file
        
    feature_list, trigger_list = subsample(feature_list, trigger_list, subsampling_rate = 0.95)    
    N_classes, N_dims = feature_list[0].shape
    N_samples = len(trigger_list)

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
            trigger = trigger_list[sample] 
            y = FV.trigger_list.index(trigger)
            
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
 
 

if 1:
    list_a = []
    list_a.append(feature_vector.phi_alternative_0)
    list_a.append(feature_vector.phi_alternative_1)
    
    FV = feature_vector.FeatureVector(list_a)
    
    if 0:
        listOfFiles = utils.list_files()
        f1 = utils.load_json_file(listOfFiles[0])
        sentence = f1['sentences'][0]   #pick first sentence
        token_index = 0 #first word in sentence
        feature_matrix = FV.get_feature_matrix(token_index, sentence)
    
    
    train,valid = utils.create_training_and_validation_file_lists(ratio = 0.75, load=True)    

    Lambda, misclassification_rates = train_perceptron(FV, train[:20], T_max = 25, LR = 10.0)   
    plt.plot(misclassification_rates)    
    
    (y_hat, y) = test_perceptron(FV, Lambda, valid[:3])
    
    
    errors = [1 for y1,y2 in zip(y_hat, y) if y1!=y2]
    misclassification_rate = len(errors)/float(len(y))
    
    
    
    
    
    
    
 