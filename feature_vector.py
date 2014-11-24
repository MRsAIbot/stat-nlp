"""
Class template for feature vectors. f(x,c)
Extending the dictionary class to return 0 if one tries to acceess a feature vector for a missing key.
An alternative implementation is commented below in case the method of extending the dict class is not adequate.
"""
class FeatureVector(dict):
    # this extended dictionary class is initialised by passing a list of functions to it. These are then assigned as dictionary items upon init.
    def __init__(self, phi_list):
        i=0
        for phi in phi_list:
            self[i]=phi
            i+=1

    # returns 0 if trying to access a non-existing feature function
    def __missing__(self, key):
        return 0

    # returns a vector with all the feature functions evaluated at the given inputs (word, c) as elements
    def get_vector(self, word, c):
        phi_vector = []
        for phi in self:
            phi_vector.append(self[phi](word,c))
        return phi_vector

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
