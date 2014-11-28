import numpy as np
from scipy.sparse import issparse

class NaiveBayes(object):
	"""docstring for NaiveBayes"""
	def __init__(self, fit_prior=None, class_prior=None):
		self.fit_prior = fit_prior
		self.class_prior = class_prior


	def safe_sparse_dot(a, b, dense_output=False):
		if issparse(a) or issparse(b):
			ret = a * b
			if dense_output and hasattr(ret, "toarray"):
				ret = ret.toarray()
			return ret

	# An internal function to compute the log likelihood
	def _joint_log_likelihood(self, X):
		X = check_array(X, accept_sparse='csr')
		return (safe_sparse_dot(X, self.feature_log_prob_.T) + self.class_log_prior_)

	# A function that fits our model
	def fit(self, X, y):
		pass

	# A function to partially fit our model to enable online learning
	def fit_online(self, X, y):
		pass

	def _count(self, X, y):
		self.feature_count = safe_sparse_dot(y.T, X)
		self.class_count = y.sum(axis=0)

	# Apply add-K smoothing
	def _apply_smoothing(self, k=1.0):
		feature_count_smooth = self.feature_count + a
		class_count_smooth = feature_count_smooth.sum(axis=1)

		self.feature_log_prob_ = np.log(feature_count_smooth) - np.log(class_count_smooth.reshape(-1,1))

	# Given a new set of datapoints, this function predict the class
	def predict(self, X):
		joint_log_likelihood = self._joint_log_likelihood(X)
		return self.classes_[np.argmax(joint_log_likelihood, axis=1)]

	def precision(self, X):
		pass

	def recall(self, X):
		pass

	def f1_measure(self, X):
		pass

	def accuracy(self, X):
		pass