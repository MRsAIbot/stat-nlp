import numpy as np
from scipy.sparse import issparse
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import coo_matrix


class NaiveBayes(object):
	def __init__(self, k=1.0):
		self.k = k
		# self.classes = []
		# self.feature_log_prob = []
		self.class_prior = []

	def train(self, X, y):
		n_features = X.shape[1]

		# class_prior = self.class_prior

		# Binarize Y
		labelbin = LabelBinarizer()
		Y = labelbin.fit_transform(y)
		self.classes = labelbin.classes_
		if Y.shape[1] == 1:
			Y = np.concatenate((1 - Y, Y), axis=1)

		n_effective_classes = Y.shape[1]
		self.class_count = np.zeros(n_effective_classes)
		self.feature_count = np.zeros((n_effective_classes, n_features))

		self.class_count = Y.sum(axis=0)
		self.feature_count = np.dot(Y.T, X)

		# Apply add-k-smoothing
		self.class_count_smooth = self.class_count + self.k * len(self.classes)
		self.feature_count_smooth = self.feature_count + self.k

		# Convert to log probabilities
		self.feature_log_prob = (np.log(self.feature_count_smooth) - np.log(self.class_count_smooth.reshape(-1,1)))
		self.class_log_prior = np.zeros(len(self.classes)) - np.log(len(self.classes))

		return self

	def predict(self, X):
		neg_prob = np.log(1 - np.exp(self.feature_log_prob))
		jll = np.dot(X, (self.feature_log_prob - neg_prob).T)
		jll += self.class_log_prior + neg_prob.sum(axis=1)
		return self.classes[np.argmax(jll, axis=1)]

	def evaluate(self, X, y):
		# Construct confusion matrix
		y_true = y
		y_pred = self.predict(X)

		labels = np.unique(y_true)

		n_labels = labels.size
		label_to_ind = dict((y, x) for x, y in enumerate(labels))
		# convert yt, yp into index
		y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
		y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])

		# intersect y_pred, y_true with labels, eliminate items not in labels
		ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
		y_pred = y_pred[ind]
		y_true = y_true[ind]

		CM = coo_matrix((np.ones(y_true.shape[0]), (y_true, y_pred)), shape=(n_labels, n_labels)).toarray()

		# Compute Recall
		recall_denom = np.sum(CM, axis=1)
		diag = CM.diagonal()
		# Macro-average
		recall = np.mean(np.divide(diag, recall_denom))

		# Compute precision
		prec_denom = np.sum(CM, axis=0)
		precision = np.mean(np.divide(diag,prec_denom))

		# F1 measure (F-score with beta=1)
		F1 = 2*precision*recall/(precision+recall)

		# Macro-average
		return CM, precision, recall, F1




