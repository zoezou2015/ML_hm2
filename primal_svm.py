from __future__ import division
import numpy 
from math import exp 

#import matplotlib.pyplot as plt

import gurobipy
from gurobipy import *


#import evaluation
import predict




path1 = '/Users/Zoe/documents/PhD/Semester Jan-Apr 2015/Machine Learning/HW2/data/A/train.cvs'
path2 = '/Users/Zoe/documents/PhD/Semester Jan-Apr 2015/Machine Learning/HW2/data/A/test.cvs'

train_dataSet = []  
train_labels = []  
fileIn1 = open(path1)  
for line in fileIn1.readlines():  
	lineArr = line.strip().split(' ')  
	print type(lineArr[2])
	train_dataSet.append([float(lineArr[0]), float(lineArr[1])])  
	train_labels.append(int(float(lineArr[2])))

test_dataSet = []  
test_labels = [ ]  
fileIn2 = open(path2)  
for line in fileIn2.readlines():  
	lineArr = line.strip().split(' ')  
	print type(lineArr[2])
	test_dataSet.append([float(lineArr[0]), float(lineArr[1])])  
	test_labels.append(int(float(lineArr[2])))
print len(train_labels)





def primal_svm(train_dataSet, train_labels, test_dataSet, test_labels, C):
	


	train_sample_num = len(train_labels)
	train_feature_num = len(train_dataSet[0])
	test_sample_num = len(test_labels)
	test_feature_nume = len(test_dataSet[0])

	# Create a new model
	m = Model("qp")
	m.setParam('OutputFlag', False) 

	# Create variables
	lamda = 1.0 / C
	w = [m.addVar(lb = -GRB.INFINITY, name = 'w' + str(i+1)) for i in range(train_feature_num)]
	b = m.addVar(lb = -GRB.INFINITY, name = 'b')
	kathe = [m.addVar(lb = 0, name = 'kathe' + str(i+1)) for i in range(train_sample_num)]

	# Integrate new variables
	m.update()

	# Set objective: primal form
	obj = lamda * numpy.inner(w, w) / 2.0 +  numpy.sum(kathe) 
	m.setObjective(obj)

	# Add constraint
	for i in range(train_sample_num):
		m.addConstr(numpy.multiply(train_labels[i], numpy.inner(w, train_dataSet[i]) + b) >= 1 - kathe[i])
		#m.addConstr(kathe[i] >= 0)

	
	m.optimize()

	#get parameters

	w = [i.x for i in w]
	b = b.x
	kathe = [i.x for i in kathe]

	margin = 2 / numpy.sqrt(numpy.inner(w, w))
	sv = 0
	for i in range(train_sample_num):
		if  abs(numpy.inner(w, train_dataSet[i]) + b) <= 1 :
			sv += 1
	
	print 'support vectors: ',sv
	print 'margin: ',margin

	predict.predict(train_dataSet, train_labels, test_dataSet, test_labels, w, b)

	#return w, b, kathe


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

#CC = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
CC = [1.0]
for C in CC :
	primal_svm(train_dataSet, train_labels, test_dataSet, test_labels, C)
#[train_predict_cat, test_predict_cat] = predict.predict(train_dataSet, train_labels, test_dataSet, test_labels, w1, b1)
#showSVM[train_dataSet,train_labels,test_dataSet,test_labels]
