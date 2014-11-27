"""
@author: Tobias Rijken
"""
import numpy as np
from scipy.sparse import issparse

class NaiveBayes(object):
	"""docstring for NaiveBayes"""
	def __init__(self, arg):
		super(NaiveBayes, self).__init__()
		self.arg = arg

	def safe_sparse_dot(a, b, dense_output=False):
		if issparse(a) or issparse(b):
			ret = a * b
			if dense_output and hasattr(ret, "toarray"):
				ret = ret.toarray()
			return ret

	def _joint_log_likelihood(self, X):
		X = check_array(X, accept_sparse='csr')
		return (safe_sparse_dot(X, self.feature_log_prob_.T) + self.class_log_prior_)

	def fit(self, X, y):
		pass

	def fit_online(self, X, y):
		pass

	def _count(self, X, y):
		self.feature_count = safe_sparse_dot(y.T, X)
		self.class_count = y.sum(axis=0)

	def predict(self, X):
		joint_log_likelihood = self._joint_log_likelihood(X)
		return self.classes_[np.argmax(joint_log_likelihood, axis=1)]