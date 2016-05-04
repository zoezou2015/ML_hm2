
import numpy 
from math import exp 

#import matplotlib.pyplot as plt

import gurobipy
from gurobipy import *


import evaluation
import predict




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



def dual_svm(train_dataSet, train_labels, test_dataSet, test_labels, C):
	
	train_sample_num = len(train_labels)
	train_feature_num = len(train_dataSet[0])
	test_sample_num = len(test_labels)
	test_feature_num = len(test_dataSet[0])

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
			obj_sum += alpha[i] * alpha [j] * train_labels[i] * train_labels[j] * numpy.inner(train_dataSet[i], train_dataSet[j])

	obj =  numpy.sum(alpha) - obj_sum / 2.0

	m.setObjective(obj, GRB.MAXIMIZE)

	# Add constraint
	sample_sum = 0
	for i in range(train_sample_num):
		sample_sum += alpha[i] * train_labels[i]
		m.addConstr(alpha[i] <= C)
	m.addConstr(sample_sum == 0)

	m.optimize()

	#get parameter
	alpha = [i.x for i in alpha]
	

		

	[w, b] = estimate(train_dataSet, train_labels, alpha, C)
	predict.predict(train_dataSet, train_labels, test_dataSet, test_labels, w, b)
	#return opt_alpha
	#return w, b

	
def estimate(train_dataSet, train_labels, alpha, C):

	w = numpy.zeros(len(train_dataSet[0]))
	temp_b = []
	b = 0
	for i in range(len(train_labels)):
		w = numpy.add(w, numpy.multiply(alpha[i] * train_labels[i], train_dataSet[i]))

	for i in range(len(train_labels)):
		if alpha[i] < C:
			temp = train_labels[i] - numpy.inner(w, train_dataSet[i])
			temp_b.append(temp)
	b = numpy.median(temp_b)

	#print w, b
	return w,b



#def showSVM(train_dataSet, train_labels, test_dataSet, test_labels):
	

	#if train_feature_num != 2:  
	#	print "Sorry! I can not draw because the dimension of your data is not 2!"  
	#	return 1  
  
	# draw all samples  
	#for i in range(train_sample_num): 
		#if train_labels[i] == -1:  
			#plot(train_dataSet[i, 0], train_dataSet[i, 1], 'or')  
		#elif train_labels[i] == 1:  
			#plot(train_dataSet[i, 0], train_dataSet[i, 1], 'ob')  
		# mark support vectors  
		#supportVectorsIndex = nonzero(svm.alphas.A > 0)[0]  
		#for i in supportVectorsIndex:  
		#    plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'oy')  
		  
		# draw the classify line  
		#w = zeros((2, 1))  
		#for i in supportVectorsIndex:  
		#    w += multiply(svm.alphas[i] * svm.train_y[i], svm.train_x[i, :].T)   
		#min_x = min(svm.train_x[:, 0])[0, 0]  
		#max_x = max(svm.train_x[:, 0])[0, 0]  
		#y_min_x = float(-svm.b - w[0] * min_x) / w[1]  
		#y_max_x = float(-svm.b - w[0] * max_x) / w[1]  
		#plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')  
		#show()  



#dual_svm(train_dataSet, train_labels, test_dataSet, test_labels, 1.0)
#[w2, b2] = dual_svm(train_dataSet, train_labels, test_dataSet, test_labels, 1.0)
#predict.predict(train_dataSet, train_labels, test_dataSet, test_labels, w2, b2)

