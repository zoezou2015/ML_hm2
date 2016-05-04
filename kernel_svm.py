from __future__ import division
import numpy
from math import exp, tanh


from gurobipy import *

import evaluation
import predict


path1 = '/Users/Zoe/documents/PhD/Semester Jan-Apr 2015/Machine Learning/HW2/data/B/train.cvs'
path2 = '/Users/Zoe/documents/PhD/Semester Jan-Apr 2015/Machine Learning/HW2/data/B/test.cvs'

train_dataSet = []  
train_labels = []  
fileIn1 = open(path1)  
for line in fileIn1.readlines():  
	lineArr = line.strip().split(' ')  
	#print type(lineArr[2])
	train_dataSet.append([float(lineArr[0]), float(lineArr[1])])  
	train_labels.append(int(float(lineArr[2])))

test_dataSet = []  
test_labels = [ ]  
fileIn2 = open(path2)  
for line in fileIn2.readlines():  
	lineArr = line.strip().split(' ')  
	#print type(lineArr[2])
	test_dataSet.append([float(lineArr[0]), float(lineArr[1])])  
	test_labels.append(int(float(lineArr[2])))
#print len(train_labels)



def kernel_svm(train_dataSet, train_labels, test_dataSet, test_labels, option,cat):
	

	if option == 1:
		print 'Linear Kernel'
	elif option == 2:
		print 'Gaussion Kernel'
	elif option == 3:
		print 'Polynomial Kernel'


	train_sample_num = len(train_labels)
	train_feature_num = len(train_dataSet[0])
	test_sample_num = len(test_labels)
	test_feature_num = len(test_dataSet[0])

	C = 1.0

	# Create a new model
	m = Model("qp")
	m.setParam('OutputFlag', False) 

	# Create variables
	#lamda = 1.0 / C
	alpha = [m.addVar(lb = 0, name = 'alpha' + str(i+1)) for i in range(train_sample_num)]
	
	
	# Integrate new variables
	m.update()

	# Set objective:dual form

	obj_sum = 0
	for i in range(train_sample_num):
		for j in range(train_sample_num):
			obj_sum += alpha[i] * alpha [j] * train_labels[i] * train_labels[j] * Kernel(train_dataSet[i], train_dataSet[j], option,cat)
		#print 'k',Kernel(train_dataSet[i], train_dataSet[j], option)


	obj = numpy.sum(alpha) - obj_sum / 2.0
	m.setObjective(obj, GRB.MAXIMIZE)

	# Add constraint
	sample_sum = 0
	for i in range(train_sample_num):
		sample_sum += alpha[i] * train_labels[i]
		m.addConstr(alpha[i] <= C)
	m.addConstr(sample_sum == 0)

	m.optimize()

	alpha = [i.x for i in alpha]

	#[w, b] = estimate(train_dataSet, train_labels, alpha, C)
	b = estimate(train_dataSet, train_labels, alpha, C,option,cat)
	prediction(train_dataSet, train_labels, test_dataSet, test_labels, alpha, b, option,cat)
	#predict.predict(train_dataSet, train_labels, test_dataSet, test_labels, w, b)
	#print 'a', alpha
	#print 'b', b
	#return w, b

	
def estimate(train_dataSet, train_labels, alpha, C, option,cat):

	#w = numpy.zeros(len(train_dataSet[0]))
	temp_b = []
	b = 0
	#for i in range(len(train_labels)):
	#	w = numpy.add(w, numpy.multiply(alpha[i] * train_labels[i], train_dataSet[i]))

	for i in range(len(train_labels)):
		if alpha[i] < C and alpha[i] > 0:
			temp = 0
			for j in range(len(train_labels)):
				temp += alpha[j] * train_labels[j] * Kernel(train_dataSet[j], train_dataSet[i],option,cat)
			temp = train_labels[i] - temp
			temp_b.append(temp)
	b = numpy.median(temp_b)

	#print w, b
	#return w,b
	return b


def prediction(train_dataSet, train_labels, test_dataSet, test_labels, alpha, b, option,cat):
	
	train_predict_cat = numpy.zeros(len(train_labels))
	test_predict_cat = numpy.zeros(len(test_labels))

	for i in range (len(train_labels)):
		temp = 0
		for j in range(len(train_labels)):
			temp += alpha[j] * train_labels[j] * Kernel(train_dataSet[j], train_dataSet[i], option,cat)
		#print 't', temp	
		#print 'b', temp + b
		if (temp + b) < 0:
			train_predict_cat[i] = -1
		else:
			train_predict_cat[i] = 1
			#print train_predict_cat[i]
	#print numpy.sum(train_predict_cat)
	print 'Accuracy for training dataset: ',evaluation.evaluation(train_predict_cat, train_labels)

	for i in range (len(test_labels)):
		temp = 0
		for j in range(len(train_labels)):
			temp += alpha[j] * test_labels[j] * Kernel(train_dataSet[j], test_dataSet[i], option,cat)
		if (temp + b) < 0:
			test_predict_cat[i] = -1
		else:
			test_predict_cat[i] = 1
	print 'Accuracy for test dataset: ',evaluation.evaluation(test_predict_cat, test_labels)



def Kernel(x1, x2, option,cat):
	x1 = numpy.array(x1)
	x2 = numpy.array(x2)
	if option == 1:
		# linear kernel 
		return numpy.inner(x1,x2)
	elif option == 2:
		# RBF kernel
		if cat == 'A':
			gamma = 5
		elif cat == 'B':
			gamma = 0.5
		elif cat == 'C':
			gamma = 5
		#numpy.inner(x1 - x2, x1 - x2)
		return numpy.exp(-numpy.inner(x1 - x2, x1 - x2) * gamma)

	elif option == 3:
		#polynomial kernel
		if cat == 'A':
			a = 1.138
			c = 3
			d = 2
		elif cat == 'B':			
			a = 0.98
			c= 0
			d = 1
		elif cat == 'C':	
			a = 1.138
			c = 3
			d = 2
		#d = 1
		return (numpy.inner(numpy.multiply(a, x1), x2) + c)**d


#[w, b, kathe] = primal_svm(train_dataSet, train_labels, test_dataSet, test_labels, 1.0)
#predict.predict(train_dataSet, train_labels, test_dataSet, test_labels, w, b)

#kernel_svm(train_dataSet, train_labels, test_dataSet, test_labels, 2,'B')

