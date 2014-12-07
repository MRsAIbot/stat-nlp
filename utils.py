# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 19:07:27 2014

This file contains a general collection of function, used by the other files.

@author: Johannes
"""
import glob
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import numpy as np
import cPickle
import warnings


# Create a list of .json file names
def list_files(path="./bionlp2011genia-train-clean/*.json"):
	return glob.glob(path)


# Opens and loads a single json file name and returns it
def load_json_file(file_name):
	with open(file_name) as raw_json:
		d = json.load(raw_json)
		return d


# compute precision, treating the none event differently
def precision(Confusion_matrix, None_label):
    numerator = np.sum(Confusion_matrix.diagonal()) - Confusion_matrix[None_label,None_label]
    denominator = np.sum(np.delete(Confusion_matrix, None_label, 1))
    precision = numerator/float(denominator)
    return precision


# compute recall, treating the none event differently
def recall(Confusion_matrix, None_label):
    numerator = np.sum(Confusion_matrix.diagonal()) - Confusion_matrix[None_label,None_label]
    denominator = np.sum(np.delete(Confusion_matrix, None_label, 0))
    recall = numerator/float(denominator) 
    return recall
    
    
# generic evaluation function for predictions.    
def evaluate(y_gold, y_pred, FV, mode):
    if mode == 'Trigger':
        C = len(FV.trigger_list)
        None_label = FV.trigger_list.index(u'None')
    elif mode == 'Arguments':
        C = len(FV.arguments_list)
        None_label = FV.arguments_list.index(u'None')
    else:
        warnings.warn('ERROR in get_confusion_matrix(): must use correct mode.')

    Confusion_matrix = get_confusion_matrix(y_gold, y_pred, C)
    p = precision(Confusion_matrix, None_label)
    r = recall(Confusion_matrix, None_label)
    F1 = 2 * p*r / (p+r)
    print "Precision:", p
    print "Recall:", r
    print "F1-score:", F1
    return p,r,F1
    
    
# computes a confusion matrix for gold and predicted labels. C = #classes
def get_confusion_matrix(y_gold, y_pred, C):
    Confusion = np.zeros([C, C], dtype = np.int64)
    
    for gold in range(C):
        for pred in range(C):
            gold_indices = set(np.where(np.asarray(y_gold) == gold)[0]) 
            pred_indices = set(np.where(np.asarray(y_pred) == pred)[0])
            
            s = gold_indices.intersection(pred_indices)
            Confusion[gold, pred] = len(s)
    print Confusion
    return Confusion


# Returns a dictionary with a count for all triggers
def get_all_triggers(file_list):
	trigger_dict = defaultdict(int)
	for f in file_list:
		f_json = load_json_file(f)
		for i in range(len(f_json['sentences'])):
			trigger_list = f_json['sentences'][i]['eventCandidates']
			for trigger in trigger_list:
				trigger_dict[trigger['gold']] += 1

	return trigger_dict
 
 
# Returns a list with all triggers encountered in the data set.
def get_trigger_list(load = True):
    if load:
        with open('trigger_list.data', 'rb') as f:
            loaded = cPickle.load(f)
        trigger_list = correct_end_of_lines_in_saved_list(loaded)

    else:
        file_list = list_files()
        trigger_dict = get_all_triggers(file_list)
        trigger_list = list(trigger_dict)
        with open('trigger_list.data', 'wb') as f:
            cPickle.dump(trigger_list, f)
    return trigger_list
     
 
# Returns a dictionary with a count for all argument labels [None, Theme, Cause]
def get_all_arguments(file_list):
    argument_dict = defaultdict(int)
    for f in file_list:
        f_json = load_json_file(f)
        for sentence in f_json['sentences']:
            event_candidates_list = sentence['eventCandidates']
            for event_candidates in event_candidates_list:
                arguments_list = event_candidates['arguments']
                
                for argument in arguments_list:
                    argument_dict[argument['gold']] += 1
    return argument_dict
    

# return a list of all part of speech tags encountered in the data set.
# grammar tags in this context are pos tags.
def identify_all_grammar_tags(file_list):   
    grammar_dict = defaultdict(int)
    for f in file_list:
        f_json = load_json_file(f)
        for sentence in f_json['sentences']:
            for token in sentence['tokens']:
                grammar_dict[token['pos']] +=1
    if 0:   #make nice plot
        counts = [grammar_dict[key] for key in grammar_dict.keys()]
        str_keys = [str(key) for key in grammar_dict.keys()]
        
        #visualise result of grammar_dict
        plt.figure()
        plt.title('All observed grammar tags and their frequency')
        plt.stem(counts)
        plt.xlabel('Grammar Tag')
        plt.ylabel('Total Occurrence frequency')
        plt.xticks(range(len(str_keys)), str_keys , rotation=90)
    return grammar_dict


# identifying all part-of-speech (pos)tags, which we here name grammar tags.
def get_grammar_tag_list(load = True):
    if load:
        with open('grammar_tags_list.data', 'rb') as f:
            loaded = cPickle.load(f)
        gt_list = correct_end_of_lines_in_saved_list(loaded)

    else:
        file_list = list_files()
        gt = identify_all_grammar_tags(file_list)
        gt_list = list(gt)
        with open('grammar_tags_list.data', 'wb') as f:
            cPickle.dump(gt_list, f)
    return gt_list
    
    
# identifying argument word stems
def identify_typical_argument_word_stems():
    file_list = list_files()
    stem_dict = defaultdict(int)
    arguments = list(get_all_arguments(file_list) )
    for arg in arguments:
        stem_dict[arg] = defaultdict(int)
        
    for i_f,f in enumerate(file_list):
        print 'Stem_list_arg:', i_f ,"of" ,len(file_list)
        f_json = load_json_file(f)
        for sentence in f_json['sentences']:
            eventCandidates = sentence['eventCandidates']
            for ec in eventCandidates:
                arguments_list = ec['arguments']
                for argument in arguments_list:
                    index = argument['begin'] 
                    stem = sentence['tokens'][index]['stem']
                    stem_dict[argument['gold']][stem] +=1
    return stem_dict    
    
    
# create list of most common argument word stems
def create_stem_list_arguments(cutoff = 5, load = True):
    if load == True:
        print ('Loading stem-list from file.')
        with open('stem_list_arguments.data', 'rb') as f:
            loaded = cPickle.load(f)
        stem_list = correct_end_of_lines_in_saved_list(loaded)
    else:
        print ('Computing stem-list')
        sd = identify_typical_argument_word_stems()
        stem_list = []
        for key in sd.keys()[1:]:
            counts = sd[key]
            for ckey in counts.keys():
                if counts[ckey] > cutoff:
                    stem_list += [ckey]
        
        #get rid of double elements
        stem_list = list(set(stem_list))    
        #save to file.    
        with open('stem_list_arguments.data', 'wb') as f:
            cPickle.dump(stem_list, f)
    return stem_list


# identify the most common word stems of trigger events.
def identify_typical_trigger_word_stems():
    file_list = list_files()
    stem_dict = defaultdict(int)
    triggers = list(get_all_triggers(file_list) )
    for trigger in triggers:
        stem_dict[trigger] = defaultdict(int)
        
    for f in file_list:
        f_json = load_json_file(f)
        for sentence in f_json['sentences']:
            eventCandidates = sentence['eventCandidates']
            for ec in eventCandidates:
                index = ec['begin'] 
                stem = sentence['tokens'][index]['stem']
                stem_dict[ec['gold']][stem] +=1
    return stem_dict


# identify all dependency labels that occurr in the whole data set.
def identify_all_dep_labels(return_as_list = True, load = True):
    if load:
        with open('dep_list_total.data', 'rb') as f:
            loaded = cPickle.load(f)
        key_list = correct_end_of_lines_in_saved_list(loaded)
        return key_list
    else:
        dep_dict = defaultdict(int)
        file_list = list_files()
        for i_f, f in enumerate(file_list):
            print i_f, 'of', len(file_list), 'identifying all deps'
            f_json = load_json_file(f)
            for sentence in f_json['sentences']:
                for dep in sentence['deps']:
                    dep_dict[dep['label']] +=1
        key_list = dep_dict.keys()
        #save data
        with open('dep_list_total.data', 'wb') as f:
            cPickle.dump(key_list, f)
    if return_as_list:
        return key_list
    else:
        return dep_dict
 
 
# identify most common trigger to argument dependencies
def identify_typical_trigger_argument_deps():
    file_list = list_files()
    dep_dict = defaultdict(int)
    triggers = list(get_all_triggers(file_list) )
    for trigger in triggers:
        dep_dict[trigger] = defaultdict(int)
        
    for i_f, f in enumerate(file_list):
        print i_f
        f_json = load_json_file(f)
        for sentence in f_json['sentences']:
            eventCandidates = sentence['eventCandidates']
            for ec in eventCandidates:
                index_trigger = ec['begin']
                for argument in ec['arguments']:
                    index_argument = argument['begin']
                    for dep in sentence['deps']:
                        if dep['head'] == index_trigger and dep['mod']==index_argument:
                            mod = dep['label']
                            dep_dict[ec['gold']][mod] +=1
    return dep_dict


# return most common trigger to argument dependencies
def create_dep_list_trig2arg(cutoff = 2, load = False):
    if load:
        with open('trig2arg_deps.data', 'rb') as f:
            loaded = cPickle.load(f)
        trig2arg_deps = correct_end_of_lines_in_saved_list(loaded)
        return trig2arg_deps
    else:
        trig2arg_deps = []
        dep_dict = identify_typical_trigger_argument_deps()
        for trigger in dep_dict.keys():
            for dep in dep_dict[trigger].keys():
                if dep_dict[trigger][dep] > cutoff:
                    trig2arg_deps += [dep]
        #remove double entries + save
        trig2arg_deps = list(set(trig2arg_deps))
        with open('trig2arg_deps.data', 'wb') as f:
            cPickle.dump(trig2arg_deps, f)
        return trig2arg_deps


# identify most common trigger stems, return them in list
def create_stem_list_trigger(cutoff = 5, load = True):
    if load == True:
        print ('Loading stem-list from file.')
        with open('stem_list_trigger.data', 'rb') as f:
            loaded = cPickle.load(f)
        stem_list = correct_end_of_lines_in_saved_list(loaded)
    else:
        print ('Computing stem-list')
        sd = identify_typical_trigger_word_stems()
        stem_list = []
        for key in sd.keys()[1:]:
            counts = sd[key]
            for ckey in counts.keys():
                if counts[ckey] > cutoff:
                    stem_list += [ckey]
        
        #get rid of double elements
        stem_list = list(set(stem_list))    
        #save to file.    
        with open('stem_list_trigger.data', 'wb') as f:
            cPickle.dump(stem_list, f)
    return stem_list


#computes random permutation of files and splits it into training and test set.
def create_training_and_validation_file_lists(ratio = 0.75, load = True):
    #ratio determines the ratio between training and validation set size
    if load == True:    #load previously saved splitting into train-valid sets
        print 'Loading predetermined training/validation splitting from file.'
        f = open('training_validation_files',"rb")
        t,v = cPickle.load(f)
        return t,v

    else:
        all_files = list_files()
        L = len(all_files)
        
        perm = np.random.permutation(L)
        split_index = np.int(np.floor(L*ratio))
        
        training_files   = [all_files[p] for p in perm[ :split_index] ]
        validation_files = [all_files[p] for p in perm[split_index: ] ]
    
        #save to file.
        savedata = (training_files, validation_files)
        f = open('training_validation_files',"w")
        cPickle.dump(savedata, f)
        f.close()
    return training_files, validation_files
    
   
# function to remove automatically appended \r line endings 
def correct_end_of_lines_in_saved_list(input_list):
    output_list = []
    for element in input_list:
        if '\r' in element:
            output_list += [element.strip('\r')  ]
        else:
            output_list += [element]  
    return output_list
    

# creates a dictionary with for each trigger label: a list of words and counts
# of how often this word is head of a dependency with the trigger as mod.
def identify_typical_trigger_word_mods(return_as_list = True, load = True):
    if load:
        with open('mod_list_total.data', 'rb') as f:
            loaded = cPickle.load(f)
        key_list = correct_end_of_lines_in_saved_list(loaded)
        return key_list
    else:
        mod_dict = defaultdict(int)
        file_list = list_files()
        for i_f, f in enumerate(file_list):
            print i_f, 'of', len(file_list), 'identifying all mods'
            f_json = load_json_file(f)
            for sentence in f_json['sentences']:
                eventCandidates = sentence['eventCandidates']
                for ec in eventCandidates:
                    for dep in sentence['deps']:
                        if dep['head'] in range(ec['begin'],ec['end']+1):
                            mod = sentence['tokens'][dep['mod']]['word']
                            mod_dict[mod] +=1
        key_list = mod_dict.keys()
        #save data
        with open('mod_list_total.data', 'wb') as f:
            cPickle.dump(key_list, f)
    if return_as_list:
        return key_list
    else:
        return mod_dict    
    
    
# returns most frequent words for which there is a dependency between trigger
# and the word, in which trigger is the mod.
def create_mod_list_trigger(cutoff = 5, load = True):
    if load == True:
        print ('Loading mod-list from file.')
        with open('stem_mod_trigger.data', 'rb') as f:
            loaded = cPickle.load(f)
            mod_list = correct_end_of_lines_in_saved_list(loaded)
    else:
        print ('Computing mod-list')
        sd = identify_typical_trigger_word_mods(return_as_list=False, load=False)
        mod_list = []
        for key in sd.keys():
            counts = sd[key]
            if counts > cutoff:
                mod_list += [key]
        #get rid of double elements
        mod_list = list(set(mod_list))
        #save to file.
        with open('stem_mod_trigger.data', 'wb') as f:
            cPickle.dump(mod_list, f)
    return mod_list
    