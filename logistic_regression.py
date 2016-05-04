from __future__ import division
import numpy
import time
from random import shuffle

from numpy import *

#import sklearn.metrics as metrics

#import matplotlib.pyplot as plt


import evaluation

#path1 = '/Users/Zoe/documents/PhD/Semester Jan-Apr 2015/Machine Learning/HW2/data/A/train.cvs'
#path2 = '/Users/Zoe/documents/PhD/Semester Jan-Apr 2015/Machine Learning/HW2/data/A/test.cvs'

#train_dataSet = []  
#train_labels = []  
#fileIn1 = open(path1)  
#for line in fileIn1.readlines():  
#	lineArr = line.strip().split(' ')  
	#print type(lineArr[2])
#	train_dataSet.append([float(lineArr[0]), float(lineArr[1])])  
#	train_labels.append(int(float(lineArr[2])))

#test_dataSet = []  
#test_labels = [ ]  
#fileIn2 = open(path2)  
#for line in fileIn2.readlines():  
#	lineArr = line.strip().split(' ')  
	#print type(lineArr[2])
#	test_dataSet.append([float(lineArr[0]), float(lineArr[1])])  
#	test_labels.append(int(float(lineArr[2])))
#print len(train_labels)



def logistic_reg(train_dataSet, train_labels, test_dataSet, test_labels,lamda):
	
	#print '--------------------- Logistic Regression ----------------'
	#print 'Loading data...'
	#load data
	#[x, y, train_size,x_test,y_test,test_size] = document_vectorize.createDataSet(train_path, test_path, category, k)

	MaxIteration = 100
	
	train_sample_num = len(train_labels)
	test_sample_num = len(test_labels)
	feature_num =  2
	


	#lamdas = [0.0001,0.001, 0.01, 0.1, 1, 2, 5, 6,10,100,1000]
   	#lamdas = [100]

	#for lamda in lamdas:

	update_loss = 0
	min_loss = 'Inf'
	train_predict_cat = numpy.zeros(train_sample_num)
	test_predict_cat = numpy.zeros(test_sample_num)

	w = numpy.zeros(feature_num)
	b = 0
	min_w = numpy.zeros(feature_num)  
	min_b = 0
	#print '------------------------------------------------------------'
	#print 'lamda = ', lamda

	shuffle_order = range(train_sample_num)
	#start traing
	start_time = time.time()
	for iteration in range(MaxIteration):

		learn_rate = 1/(iteration+1)
		shuffle(shuffle_order)

		#stochastic gradient descent
		for t in shuffle_order:
			
			temp1 = numpy.add(numpy.inner(w, train_dataSet[t]), b)
			temp2 = numpy.exp(numpy.multiply(temp1, train_labels[t]))
			temp3 = learn_rate / (1 + temp2) 
			w = numpy.add( (1 - lamda * learn_rate) * w, numpy.multiply(train_labels[t] * temp3, train_dataSet[t])) 
			b += learn_rate * train_labels[t] * temp3


		#print "iteration = "+str(iteration)
		#print "update_loss ="+str(update_loss)
		#print "min_loss =" +str(min_loss)

		#calculate loss
		temp_loss = 0
		for i in range (train_sample_num):
			temp1 = numpy.add(numpy.inner(w, train_dataSet[t]), b)
			temp2 = numpy.exp(numpy.multiply(-temp1, train_labels[t]))
			temp3 = 1 / (1 + temp2)
			temp_loss += numpy.log(1 + temp3)
		square = lamda * numpy.sum(numpy.square(w)) / 2.0
		update_loss = temp_loss/train_sample_num + square

		#if min_loss == 0:
		#   break
		#record minimum loss
		if min_loss > update_loss:
			min_loss = update_loss
			min_w = w
			min_b = b

		#print "min_loss = "+str(min_loss)
		#print "iteration = "+str(iteration)
		#if abs(min_loss - update_loss) < 0.000001:
		#   break
	print 'min loss', min_loss	
	#print 'Training time: ', time.time()-start_time
	print "RESULT: w: " + str(min_w) + " b: " + str(min_b)

	#predict training set
	for i in range(train_sample_num):
		if ((numpy.inner(train_dataSet[i],min_w)+min_b)) < 0:
			train_predict_cat[i] = -1
		else:
			train_predict_cat[i] = 1
	#print "# train_size = " + str(train_size)
	print 'Accuracy for training dataset: ',evaluation.evaluation(train_predict_cat, train_labels)

	#predict test set
	for i in range(test_sample_num):
		if ((numpy.inner(test_dataSet[i],min_w)+min_b)) < 0:
			test_predict_cat[i] = -1
		else:
			test_predict_cat[i] = 1
	#print "# test_size = " + str(test_size)
	print 'Accuracy for test dataset: ',evaluation.evaluation(test_predict_cat, test_labels)



#logistic_reg(train_dataSet, train_labels, test_dataSet, test_labels)

#if __name__ == "__main__":
	#training_set = document_vectorize.createDataSet(path1,category1)
	
	#pass















