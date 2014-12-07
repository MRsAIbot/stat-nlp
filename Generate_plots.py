# -*- coding: utf-8 -*-
"""
Created on Sun Dec 07 15:09:34 2014

@author: Johannes
"""
import utils
import string
import matplotlib.pyplot


if 0:
    char_list = string.printable
        
    trig_list = FV_trig.trigger_list
    character_features = Lambda_e[:,:len(char_list)]
    imshow(character_features)
    xticks(range(len(char_list)), char_list, fontsize = 8)
    colorbar(shrink=0.2)
    title('Perceptron weights of character features')
    ylabel('trigger class')
    yticks(range(len(trig_list)), trig_list, fontsize = 8)    
    xlabel('$char \in L_{char}$')
    
    
if 0:
    pos_tags = utils.get_grammar_tag_list()
    pos_features = Lambda_e[:,len(char_list):len(char_list)+len(pos_tags)]
    imshow(pos_features)
    colorbar(shrink=0.3)
    title('Perceptron weights for pos-tag features')
    ylabel('trigger class')
    xticks(range(len(pos_tags)), pos_tags, fontsize = 10, rotation=90)
    yticks(range(len(trig_list)), trig_list, fontsize = 10)    
    xlabel('pos-tag')
    
if 0:
    stem_triggers = utils.create_stem_list_trigger(cutoff = 5, load = True)
    
if 0:
    all_deps = utils.identify_all_dep_labels(load = True)
    
if 0:
    mod_list_trig = utils.create_mod_list_trigger(cutoff = 25, load=True) 

if 1:
    plot(misc2)
    title('Misclassification Rate across training epochs [argument prediction]')
    xlabel('Training Epoch')
    ylabel('misclassification rate [%]')
    grid()
       