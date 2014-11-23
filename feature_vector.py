"""
Class template for feature vectors. f(x,c)
Extending the dictionary class to return 0 if one tries to acceess a feature vector for a missing key.
An alternative implementation is commented below in case the method of extending the dict class is not adequate.
"""
class FeatureVector(dict):
    def __missing__(self, key):
        return 0

# # testing code
# feature_vector = FeatureVector()
# feature_vector[('test','Regulation')] =1
# print (feature_vector)
# # {('test', 'Regulation'): 1}
# print (feature_vector['not there'])
# # 0

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
