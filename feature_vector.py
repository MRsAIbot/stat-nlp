import assignment2 #only for testing issues (test code Johannes)
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import string
from scipy.sparse import coo_matrix

"""
Class template for feature vectors. f(x,c)
Extending the dictionary class to return 0 if one tries to acceess a feature vector for a missing key.
An alternative implementation is commented below in case the method of extending the dict class is not adequate.
"""



# Identify all possible types of grammatical objects occuring in the dataset.
# Return list of all possible objects: 'NN' 'VP', etc. --> ['NN', 'VP', ...]
#file_list = assignment2.listFiles()
def identify_all_grammar_tags(file_list):   
    grammar_dict = defaultdict(int)
    for f in file_list:
        f_json = assignment2.load_json_file(f)
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




class FeatureVector(dict):
    # this extended dictionary class is initialised by passing a list of functions to it. These are then assigned as dictionary items upon init.
    def __init__(self, phi_list):
        # also load presaved comparison lists of all grammar tags / 
        # gold label lists / argument lists that we need when evaluating the
        # phi functions. NOT DONE YET.
        i=0
        for phi in phi_list:
            self[i]=phi
            i+=1
        

    # returns 0 if trying to access a non-existing feature function
    def __missing__(self, key):
        return 0

    """
    # original function by michael
    # returns a vector with all the feature functions evaluated at the given inputs (word, c) as elements
    def get_vector(self, word, c):
        phi_vector = []
        for phi in self:
            phi_vector.append(self[phi](word,c))
        return phi_vector
    
    """
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
