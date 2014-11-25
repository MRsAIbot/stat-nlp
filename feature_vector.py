#import assignment2 #only for testing issues (test code Johannes)
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import string

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
        f_json = assignment2.loadJSONfile(f)
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

    # original function by michael
    # returns a vector with all the feature functions evaluated at the given inputs (word, c) as elements
    def get_vector(self, word, c):
        phi_vector = []
        for phi in self:
            phi_vector.append(self[phi](word,c))
        return phi_vector

    # alternative function by Johannes
    # takes as input a word + the sentence it occurs in (dict-structure from 
    # json file). 
    def get_vector_alternative(self, word, sentence):
        phi_vector = []
        for phi in self:
            phi_vector += [ self[phi](token_index,sentence) ]
        # so far phi_vector is list of lists. Flatten it to one huge list.
        phi_vector = [x for sublist in phi_vector for x in sublist]
        return phi_vector



# feature template that takes as input a token x and its sentence (which is
# a sentence from the json dictionary, containing all information about grammar
# tags, links and relations to other tokens, their positions, and finally
# also the gold labels for both triggers and arguments.) Note that the token
# is not a string, but the index at which this token appears in sentence.

# This particular function is merely an example that returns a vector full of
# indicators whether different ASCII symbols are contained within the token.
list_a = []
def phi_alternative_0(token_index, sentence):
    token = sentence['tokens'][token_index]['word']
    stem = sentence['tokens'][token_index]['stem']
    #can compute anything here: e.g. can compare token or stem with other words
    #This is merely an example for computing features across a comparison list.
    symbols_list = string.printable
    return_vec = [ np.uint8(character in token)  for character in symbols_list]
    
    return return_vec
    
    
list_a.append(phi_alternative_0)


# check for each grammar tag (NN, VP, etc.) if token is this grammatical object.
def phi_alternative_1(token_index, sentence):
    observed_grammar_tag = sentence['tokens'][token_index]['pos']    #e.g. 'NN'
    index = all_grammar_tags.index(observed_grammar_tag)
    
    unit_vec = np.zeros(len(all_grammar_tags), dtype = np.uint8)
    unit_vec[index] = 1.0
    return list(unit_vec) #or return list(unit_vec) #or return sparsified unit_vec 
list_a.append(phi_alternative_1)





# # Testing code (Johannes)
#test alternative feature functions
if 1:
    listOfFiles = assignment2.listFiles()
    f1 = assignment2.loadJSONfile(listOfFiles[0])
    sentence = f1['sentences'][0]   #pick first sentence
    token_index = 0 #first word in sentence

    grammar_dict = identify_all_grammar_tags(file_list)   
    all_grammar_tags = grammar_dict.keys()  #these lists should be saved and later loaded.
    
    f_v=FeatureVector(list_a)
    vec = f_v.get_vector_alternative( token_index, sentence)
    






"""     THIS IS ORIGINAL CODE FROM MICHAEL, UNCHANGED BY JOHANNES 24/11/14
# List of functions example (feature functions are defined and added to a list to initiate a feature_vector dictionary with -- see testing code for example)
list_a = []
def phi_0(a, b):
    if a>b:
        return 1
    else:
        return 0
list_a.append(phi_0)

def phi_1(a, b):
    if a-b==0:
        return 1
    else:
        return 0
list_a.append(phi_1)

def phi_2(a, b):
    if phi_0(a,b)==0:
        return 1
    else:
        return 0
list_a.append(phi_2)
"""




# # Testing code
# x=3
# y=3
# feature_vector = FeatureVector(list_a)
# print(feature_vector.get_vector(x,y))

"""
class FeatureVector():
    feature_vectors = {}
    def __init__(cls, word_c, weight=1.0):
        cls.feature_vectors[word_c] = weight

    @classmethod
    def get_all(cls):
        return cls.feature_vectors

# testing code
FeatureVector(('test','Regulation'))
print (FeatureVector.get_all())
# Output: {('test', 'Regulation'): 1.0}
"""
