import numpy as np
from scipy.sparse import issparse

class NaiveBayes(object):
	"""docstring for NaiveBayes"""
	def __init__(self, k=1.0, binarize=.0, fit_prior=None, class_prior=None):
		self.k = k
		self.binarize = binarize
		self.fit_prior = fit_prior
		self.class_prior = class_prior


	def safe_sparse_dot(a, b, dense_output=False):
		if issparse(a) or issparse(b):
			ret = a * b
			if dense_output and hasattr(ret, "toarray"):
				ret = ret.toarray()
			return ret

	# Apply add-K smoothing
	def _apply_smoothing(self):
		n_classes = len(self.classes_)
		feature_count_smooth = self.feature_count_ + self.k
		class_count_smooth = self.class_count_ + self.k * n_classes

		self.feature_log_prob_ = np.log(feature_count_smooth) - np.log(class_count_smooth.reshape(-1,1))

	def _update_feature_log_prob(self):
		n_classes = len(self.classes_)
		smoothed_fc = self.feature_count_ + self.alpha
		smoothed_cc = self.class_count_ + self.alpha * n_classes

		self.feature_log_prob_ = (np.log(smoothed_fc) - np.log(smoothed_cc.reshape(-1, 1)))

	# An internal function to compute the log likelihood
	def _joint_log_likelihood(self, X):
		# X = check_array(X, accept_sparse='csr')

		# if self.binarize is not None:
		# 	X = binarize(X, threshold=self.binarize)

		n_classes, n_features = self.feature_log_prob_.shape
		n_samples, n_features_X = X.shape

		if n_features_X != n_features:
			raise ValueError("Expected input with %d features, got %d instead" % (n_features, n_features_X))

		neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
		# Compute  neg_prob * (1 - X).T  as  sum(neg_prob - X * neg_prob)
		jll = safe_sparse_dot(X, (self.feature_log_prob_ - neg_prob).T)
		jll += self.class_log_prior_ + neg_prob.sum(axis=1)

		return jll

	def _update_class_log_prior(self, class_prior=None):
		n_classes = len(self.classes_)
		if class_prior is not None:
		    if len(class_prior) != n_classes:
		        raise ValueError("Number of priors must match number of classes.")
			self.class_log_prior_ = np.log(class_prior)
		elif self.fit_prior:
			# empirical prior, with sample_weight taken into account
			self.class_log_prior_ = (np.log(self.class_count_) - np.log(self.class_count_.sum()))
		else:
			self.class_log_prior_ = np.zeros(n_classes) - np.log(n_classes)

	# A function that trains our model
	def train(self, X, y, sample_weight=None):
		#X, y = check_X_y(X, y, 'csr')
		_, n_features = X.shape

		labelbin = LabelBinarizer()
		Y = labelbin.fit_transform(y)
		self.classes_ = labelbin.classes_
		if Y.shape[1] == 1:
			Y = np.concatenate((1 - Y, Y), axis=1)

		# convert to float to support sample weight consistently;
		# this means we also don't have to cast X to floating point
		Y = Y.astype(np.float64)
		if sample_weight is not None:
			Y *= check_array(sample_weight).T

		class_prior = self.class_prior

		# Count raw events from data before updating the class log prior
		# and feature log probas
		n_effective_classes = Y.shape[1]
		self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
		self.feature_count_ = np.zeros((n_effective_classes, n_features), dtype=np.float64)
		self._count(X, Y)
		self._update_feature_log_prob()
		self._update_class_log_prior(class_prior=class_prior)
		return self

	# A function to partially train our model to enable online learning
	def train_online(self, X, y):
		pass

	def _count(self, X, y):
		self.feature_count = safe_sparse_dot(y.T, X)
		self.class_count = y.sum(axis=0)

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