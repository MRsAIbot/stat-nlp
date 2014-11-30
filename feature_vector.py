import numpy as np
import matplotlib.pyplot as plt
import string
from scipy.sparse import coo_matrix
import utils

"""
Class template for feature vectors. f(x,c)
Extending the dictionary class to return 0 if one tries to acceess a feature vector for a missing key.
An alternative implementation is commented below in case the method of extending the dict class is not adequate.
"""
class FeatureVector(dict):
    # this extended dictionary class is initialised by passing a list of functions to it. These are then assigned as dictionary items upon init.
    def __init__(self, phi_list,):
        i=0
        for phi in phi_list:
            self[i]=phi
            i+=1
        self.listOfAllFiles = utils.list_files()
        self.all_grammar_tags = list(utils.identify_all_grammar_tags(self.listOfAllFiles))
        self.trigger_list = list(utils.get_all_triggers(self.listOfAllFiles) )
        


    #newest version. Finally includes features for every different class.
    def get_feature_matrix(self, token_index, sentence):
        all_col_indices = []
        all_row_indices = []
        values = []
        n_classes = 10  #length of list of all occurring triggers in dataset.
        for c in range(n_classes):
            d=0
            for phi in self:
                phi_vector = self[phi](token_index, sentence, self.all_grammar_tags)
    
                index = list(np.nonzero(np.array(phi_vector))[0])
                all_col_indices += [i+d for i in index]    # offset d in matrix                
                all_row_indices += [c]*len(index)
                
                values += list(np.array(phi_vector)[index])
                
                d += len(phi_vector)
             
        
        sparse_feature_matrix = coo_matrix((np.array(values), 
                                           (np.asarray(all_row_indices),
                                            np.array(all_col_indices) ) ),
                                            shape=(n_classes,d))
            
        return sparse_feature_matrix
        
    #Get feature matrix for argument prediction: for pairs of tokens and 
    #argument candidates.
    def get_feature_matrix_argument_prediction(self, token_index, arg_index, sentence):
        all_col_indices = []
        all_row_indices = []
        values = []
        n_classes = 3  #possible predictions: [None, Theme, Cause]
        for c in range(n_classes):
            d=0
            for phi in self:
                phi_vector = self[phi](token_index, arg_index, sentence, self.all_grammar_tags)
    
                index = list(np.nonzero(np.array(phi_vector))[0])
                all_col_indices += [i+d for i in index]    # offset d in matrix                
                all_row_indices += [c]*len(index)
                
                values += list(np.array(phi_vector)[index])
                
                d += len(phi_vector)
             
        
        sparse_feature_matrix = coo_matrix((np.array(values), 
                                           (np.asarray(all_row_indices),
                                            np.array(all_col_indices) ) ),
                                            shape=(n_classes,d))
            
        return sparse_feature_matrix



# feature template that takes as input a token x and its sentence (which is
# a sentence from the json dictionary, containing all information about grammar
# tags, links and relations to other tokens, their positions, and finally
# also the gold labels for both triggers and arguments.) Note that the token
# is not a string, but the index at which this token appears in sentence.
# This particular function is merely an example that returns a vector full of
# indicators whether different ASCII symbols are contained within the token.

def phi_alternative_0(token_index, sentence, all_grammar_tags):
    token = sentence['tokens'][token_index]['word']
    stem = sentence['tokens'][token_index]['stem']
    #can compute anything here: e.g. can compare token or stem with other words
    #This is merely an example for computing features across a comparison list.
    symbols_list = string.printable
    return_vec = [ np.uint8(character in token)  for character in symbols_list]
    
    return return_vec
    

# check for each grammar tag (NN, VP, etc.) if token is this grammatical object.
def phi_alternative_1(token_index, sentence, all_grammar_tags):
    observed_grammar_tag = sentence['tokens'][token_index]['pos']    #e.g. 'NN'
    index = all_grammar_tags.index(observed_grammar_tag)
    
    unit_vec = np.zeros(len(all_grammar_tags), dtype = np.uint8)
    unit_vec[index] = 1.0
    return list(unit_vec) #or return list(unit_vec) #or return sparsified unit_vec 


#extract if argument is a protein
def phi_argument_0(token_index, arg_index, sentence, all_grammar_tags):
    #argument = sentence['tokens'][arg_index]['word']
    protein = [0]
    for mention in sentence['mentions']:
        if arg_index >= mention['begin'] and arg_index < mention['end']:
            protein = [1]
    return protein


def phi_argument_1(token_index, arg_index, sentence, all_grammar_tags):
    observed_grammar_tag = sentence['tokens'][arg_index]['pos']    #e.g. 'NN'
    index = all_grammar_tags.index(observed_grammar_tag)
    
    unit_vec = np.zeros(len(all_grammar_tags), dtype = np.uint8)
    unit_vec[index] = 1.0
    return list(unit_vec) #or return list(unit_vec) #or return sparsified unit_vec 







""" OLD feature functions. Not thrown away yet.


# alternative function by Johannes (obsolete)
# takes as input a word + the sentence it occurs in (dict-structure from 
# json file). 
def get_vector_alternative(self, token_index, sentence, all_grammar_tags):
    all_col_indices = []
    values = []
    n_samples = 1
    d=0
    for phi in self:
        phi_vector = self[phi](token_index,sentence, all_grammar_tags)

        index = list(np.nonzero(np.array(phi_vector))[0])
        all_col_indices += [i+d for i in index] # offset d in matrix
        d += len(phi_vector)

        values += list(np.array(phi_vector)[index])

    sparse_feature_matrix = coo_matrix((np.array(values), (np.zeros(np.shape(all_col_indices)) ,np.array(all_col_indices))), shape=(n_samples,d))
    
    return sparse_feature_matrix


#New version: Compute feature vec for whole batch of inputs!
#Return a sparse matrix, rows: samples. cols: feature dimensions.
def get_feature_batch(self, token_index_list, sentence_list, all_grammar_tags):
    n_samples = len(token_index_list)
    d=0
    all_col_indices = []
    all_row_indices = []
    values = []
    for (s, (sentence, token_index)) in enumerate(zip(sentence_list, token_index_list)):
        print s
        for phi in self:
            phi_vector = self[phi](token_index,sentence, all_grammar_tags)

            index = list(np.nonzero(np.array(phi_vector))[0])
            all_col_indices += [i+d for i in index]    # offset d in matrix                
            all_row_indices += [s]*len(index)
            
            values += list(np.array(phi_vector)[index])
            
            d += len(phi_vector)
     
    
    sparse_feature_matrix = coo_matrix((np.array(values), 
                                        (np.asarray(all_row_indices),
                                        np.array(all_col_indices) ) ),
                                        shape=(n_samples,d))
        
    return sparse_feature_matrix
"""


