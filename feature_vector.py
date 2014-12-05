import numpy as np
import string
from scipy.sparse import coo_matrix
import utils
import inspect

"""
Class template for feature vectors. f(x,c)
Extending the dictionary class to return 0 if one tries to acceess a feature vector for a missing key.
An alternative implementation is commented below in case the method of extending the dict class is not adequate.
"""
class FeatureVector():
    # this extended dictionary class is initialised by passing a list of functions to it. These are then assigned as dictionary items upon init.
    def __init__(self, mode = 'argument'):
        if mode not in ['argument', 'trigger', 'joint']:
            print 'ERROR, wrong mode of calling FeatureVector class! '


        #get handles to all phi functions
        self.methods = inspect.getmembers(self, predicate=inspect.ismethod)
        self.mode = mode
        if self.mode == 'argument' or self.mode == 'trigger':
            self.phi_list = [method[1] for method in self.methods if 'phi_'+mode in method[0]]
        elif mode == 'joint':
            self.phi_list_arg = [method[1] for method in self.methods if 'phi_argument' in method[0]]
            self.phi_list_trig = [method[1] for method in self.methods if 'phi_trigger' in method[0]]


        #load relevant other data from presaved files.
        self.listOfAllFiles = utils.list_files()
        self.all_grammar_tags = utils.get_grammar_tag_list()
        self.trigger_list = utils.get_trigger_list()
        self.stem_list_triggers = utils.create_stem_list_trigger(cutoff = 5, load=True)
        self.stem_list_arguments = utils.create_stem_list_arguments(cutoff = 5, load=True)
        #self.mod_list_triggers = utils.create_mod_list_trigger(cutoff = 5)
        self.arguments_list = [u'None', u'Theme', u'Cause']
        
        self.dep_list_total = utils.identify_all_dep_labels(load = True) 
        self.trig2arg_deps = utils.create_dep_list_trig2arg(cutoff = 2, load = True)


    #Feature matrix for trigger prediction
    def get_feature_matrix(self, token_index, sentence, clf):
        """
        clf (string): 'nb' or 'perc'
        """
        all_col_indices = []
        all_row_indices = []
        values = []
        if clf == 'nb':
            n_classes = 1  #length of list of all occurring triggers in dataset.
        elif clf == 'perc':
            n_classes = 10
        for c in range(n_classes):
            d=0
            for phi in self.phi_list:
                phi_vector = phi(token_index, sentence)
    
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
    #argument candidates. Otherwise same skeleton as "get_feature_matrix()"
    def get_feature_matrix_argument_prediction(self, token_index, arg_index, sentence, clf):
        all_col_indices = []
        all_row_indices = []
        values = []
        if clf == 'nb':
            n_classes = 1
        elif clf == 'perc':
            n_classes = 3
        for c in range(n_classes):
            d=0
            for phi in self.phi_list:
                phi_vector = phi(token_index, arg_index, sentence)
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



    # feature templates take as input a token_index and sentence (which is
    # a sentence from the json dictionary, containing all information about grammar
    # tags, links and relations to other tokens, their positions, and finally
    # also the gold labels for both triggers and arguments.) Note that the token
    # is not a string, but the index at which this token appears in sentence.
    """TRIGGER FEATURES"""

    def phi_trigger_0(self, token_index, sentence):
        #character indicator
        token = sentence['tokens'][token_index]['word']
        symbols_list = string.printable
        return_vec = [ np.uint8(character in token)  for character in symbols_list]
        return return_vec

    def phi_trigger_1(self, token_index, sentence):
        #grammar (pos)-tag indicator
        observed_grammar_tag = sentence['tokens'][token_index]['pos'] #e.g. 'NN'
        index = self.all_grammar_tags.index(observed_grammar_tag)

        unit_vec = np.zeros(len(self.all_grammar_tags), dtype = np.uint8)
        unit_vec[index] = 1.0
        return list(unit_vec)

    def phi_trigger_2(self, token_index, sentence):
        #evaluate stem of token.
        observed_stem = sentence['tokens'][token_index]['stem']
        unit_vec = np.zeros(len(self.stem_list_triggers), dtype = np.uint8)

        if observed_stem in self.stem_list_triggers:
            index = self.stem_list_triggers.index(observed_stem)
            unit_vec[index] = 1.0
        return list(unit_vec)

    def phi_trigger_3(self, token_index, sentence):
        #evaluate head of token.
        dep_vec = np.zeros(len(self.dep_list_total), dtype = np.uint8)

        #return a vector with 1 for the dep_label for which the token is head.
        for dep in sentence['deps']:
            if dep['head'] == token_index:
                dep_label = dep['label']
                if dep_label in self.dep_list_total:
                    index = self.dep_list_total.index(dep_label)
                    dep_vec[index] = 1.0
        return list(dep_vec)

    def phi_trigger_4(self, token_index, sentence):
        #evaluate mod of token.
        dep_vec = np.zeros(len(self.dep_list_total), dtype = np.uint8)

        #return a vector with 1 for the dep_label for which the token is mod.
        for dep in sentence['deps']:
            if dep['mod'] == token_index:
                dep_label = dep['label']
                if dep_label in self.dep_list_total:
                    index = self.dep_list_total.index(dep_label)
                    dep_vec[index] = 1.0
        return list(dep_vec)
        
    
    #def phi_trigger_3(self, token_index, sentence):


    """ARGUMENT FEATURES"""
    def phi_argument_0(self, token_index, arg_index, sentence):
        #extract if argument is a protein   (Mentions)
        protein = [0]
        for mention in sentence['mentions']:
            if arg_index >= mention['begin'] and arg_index < mention['end']:
                protein = [1]
        return protein

    def phi_argument_1(self, token_index, arg_index, sentence):
        #evaluate grammar pos tag of argument
        observed_grammar_tag = sentence['tokens'][arg_index]['pos']
        index = self.all_grammar_tags.index(observed_grammar_tag)

        unit_vec = np.zeros(len(self.all_grammar_tags), dtype = np.uint8)
        unit_vec[index] = 1.0
        return list(unit_vec)

    def phi_argument_2(self, token_index, arg_index, sentence):
        #evaluate grammar pos tag of trigger token
        observed_grammar_tag = sentence['tokens'][token_index]['pos']
        index = self.all_grammar_tags.index(observed_grammar_tag)

        unit_vec = np.zeros(len(self.all_grammar_tags), dtype = np.uint8)
        unit_vec[index] = 1.0
        return list(unit_vec)

    def phi_argument_3(self, token_index, arg_index, sentence):
        #evaluate stem of trigger token.
        observed_stem = sentence['tokens'][token_index]['stem']
        unit_vec = np.zeros(len(self.stem_list_triggers), dtype = np.uint8)

        if observed_stem in self.stem_list_triggers:
            index = self.stem_list_triggers.index(observed_stem)
            unit_vec[index] = 1.0
        return list(unit_vec)

    def phi_argument_4(self, token_index, arg_index, sentence):
        #evaluate stem of argument token.
        observed_stem = sentence['tokens'][arg_index]['stem']
        unit_vec = np.zeros(len(self.stem_list_arguments), dtype = np.uint8)

        if observed_stem in self.stem_list_arguments:
            index = self.stem_list_arguments.index(observed_stem)
            unit_vec[index] = 1.0
        return list(unit_vec)


    def phi_argument_5(self, token_index, arg_index, sentence):
        #character indicator for argument
        token = sentence['tokens'][arg_index]['word']
        symbols_list = string.printable
        return_vec = [ np.uint8(character in token)  for character in symbols_list]
        return return_vec

    def phi_argument_6(self, token_index, arg_index, sentence):
        #evaluate head of arg_index.
        dep_vec = np.zeros(len(self.dep_list_total), dtype = np.uint8)

        #return a vector with 1 for the dep_label for which the token is head.
        for dep in sentence['deps']:
            if dep['head'] == arg_index:
                dep_label = dep['label']
                if dep_label in self.dep_list_total:
                    index = self.dep_list_total.index(dep_label)
                    dep_vec[index] = 1.0
        return list(dep_vec)

    def phi_argument_7(self, token_index, arg_index, sentence):
        #evaluate mod of arg_index.
        dep_vec = np.zeros(len(self.dep_list_total), dtype = np.uint8)

        #return a vector with 1 for the dep_label for which the token is mod.
        for dep in sentence['deps']:
            if dep['mod'] == arg_index:
                dep_label = dep['label']
                if dep_label in self.dep_list_total:
                    index = self.dep_list_total.index(dep_label)
                    dep_vec[index] = 1.0
        return list(dep_vec)
        
    def phi_argument_8(self, token_index, arg_index, sentence):
        #evaluate if trig--->arg dependency falls into one of the typical trig2arg_dep
        dep_vec = np.zeros(len(self.trig2arg_deps), dtype = np.uint8)

        #return a vector with 1 for the dep_label for which trig->arg has this label
        for dep in sentence['deps']:
            if dep['mod'] == token_index and dep['head'] == arg_index:
                dep_label = dep['label']
                if dep_label in self.trig2arg_deps:
                    index = self.trig2arg_deps.index(dep_label)
                    dep_vec[index] = 1.0
        return list(dep_vec)