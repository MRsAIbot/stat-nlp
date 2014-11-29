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


# Create a list of .json file names
def list_files(path="./bionlp2011genia-train-clean/*.json"):
	return glob.glob(path)


# Opens and loads a single json file name and returns it
def load_json_file(file_name):
	with open(file_name) as raw_json:
		d = json.load(raw_json)
		return d


# Returns a dictionary with a count of all triggers
def get_all_triggers(file_list):
	trigger_dict = defaultdict(int)
	for f in file_list:
		f_json = load_json_file(f)
		for i in range(len(f_json['sentences'])):
			trigger_list = f_json['sentences'][i]['eventCandidates']
			for trigger in trigger_list:
				trigger_dict[trigger['gold']] += 1

	return trigger_dict
 
#Returns a dictionary with a count of all arguments (=labels of the relations)
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
    



# Identify all possible types of grammatical objects occuring in the dataset.
# Return list of all possible objects: 'NN' 'VP', etc. --> ['NN', 'VP', ...]
#file_list = assignment2.listFiles()
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
    
    
    
    
    
    
    
    
    
    
    
    
    