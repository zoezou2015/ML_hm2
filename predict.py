from __future__ import division
import numpy 
from numpy import *
#import matplotlib.pyplot as plt   



import evaluation

def predict(train_dataSet, train_labels, test_dataSet, test_labels, w, b):
	"""
	Predict training and test set

	"""
	train_sample_num = len(train_labels)
	train_feature_num = len(train_dataSet[0])
	test_sample_num = len(test_labels)
	test_feature_num = len(test_dataSet[0])

	train_predict_cat = numpy.zeros(train_sample_num)
	test_predict_cat = numpy.zeros(test_sample_num)

	#predict training set
	for i in range(train_sample_num):
	  if ((numpy.inner(train_dataSet[i],w)+b)) <= 0:
		#print numpy.inner(train_dataSet[i],w)+b
		train_predict_cat[i] = -1
	  else:
	  	#print numpy.inner(train_dataSet[i],w)+b
		train_predict_cat[i] = 1
	#print "# train_size = " + str(train_size)
	print 'Accuracy for training dataset: ',evaluation.evaluation(train_predict_cat, train_labels)

	#predict test set
	for i in range(test_sample_num):
	  if ((numpy.inner(test_dataSet[i], w) + b)) <= 0:
	  	#print numpy.inner(test_dataSet[i],w)+b
		test_predict_cat[i] = -1
	  else:
		test_predict_cat[i] = 1
	#print "# test_size = " + str(test_size)
	print 'Accuracy for test dataset: ',evaluation.evaluation(test_predict_cat, test_labels)
	return train_predict_cat, test_predict_cat