# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 16:33:56 2014

@author: Johannes
"""

#sd = identify_typical_trigger_word_stems()

stem_list = []
zz=0
for key in sd.keys()[1:]:
    counts = sd[key]
    for ckey in counts.keys():
        if counts[ckey] > 5:
            print key, ckey
            stem_list += [ckey]
            zz +=1

stem_list = list(set(stem_list))




class F_test():
    def __init__(self, phi_list = [0]):
        i=0
        for phi in phi_list:
            self[i]=phi
            
    mm=[method for method in dir(FV_arg) ]
    phi_list = [method for method in mm if 'all' in method]

    def get_feature_matrix_argument_prediction(self, token_index, arg_index, sentence):
        pass

    def phi_alternative_0(self, token_index, sentence):
        token = sentence['tokens'][token_index]['word']
        stem = sentence['tokens'][token_index]['stem']
        #can compute anything here: e.g. can compare token or stem with other words
        #This is merely an example for computing features across a comparison list.
        symbols_list = string.printable
        return_vec = [ np.uint8(character in token)  for character in symbols_list]
        
        return return_vec