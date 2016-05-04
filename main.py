from __future__ import division
import numpy
import time
from random import shuffle

from numpy import *



import evaluation
import logistic_regression
import primal_svm
import kernel_svm
import dual_svm


path_A_train = '/Users/Zoe/documents/PhD/Semester Jan-Apr 2015/Machine Learning/HW2/data/A/train.cvs'
path_A_test = '/Users/Zoe/documents/PhD/Semester Jan-Apr 2015/Machine Learning/HW2/data/A/test.cvs'
path_B_train = '/Users/Zoe/documents/PhD/Semester Jan-Apr 2015/Machine Learning/HW2/data/B/train.cvs'
path_B_test = '/Users/Zoe/documents/PhD/Semester Jan-Apr 2015/Machine Learning/HW2/data/B/test.cvs'
path_C_train = '/Users/Zoe/documents/PhD/Semester Jan-Apr 2015/Machine Learning/HW2/data/C/train.cvs'
path_C_test = '/Users/Zoe/documents/PhD/Semester Jan-Apr 2015/Machine Learning/HW2/data/C/test.cvs'


train_path = [path_A_train, path_B_train, path_C_train] 
test_path = [path_A_test, path_B_test, path_C_test]
Cat = ['A', 'B', 'C']

def main(train_path, test_path):

	for i in range(len(train_path)):
		train_dataSet = []  
		train_labels = []  
		fileIn1 = open(train_path[i])  
		for line1 in fileIn1.readlines():  
			line1Arr = line1.strip().split(' ')  
			#print type(lineArr[2])
			train_dataSet.append([float(line1Arr[0]), float(line1Arr[1])])  
			train_labels.append(int(float(line1Arr[2])))

		test_dataSet = []  
		test_labels = [ ]  
		fileIn2 = open(test_path[i])  
		for line2 in fileIn2.readlines():  
			line2Arr = line2.strip().split(' ')  
			#print type(lineArr[2])
			test_dataSet.append([float(line2Arr[0]), float(line2Arr[1])])  
			test_labels.append(int(float(line2Arr[2])))

		print 'Classifying data', Cat[i]

		print '**********Question 1***************'
		CC = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
		for C in CC:
			print '-------------------------------------------------------------------'
			print 'C: ',C
			print 'primal form'
			primal_svm.primal_svm(train_dataSet, train_labels, test_dataSet, test_labels,C)
			print 'dual form'
			dual_svm.dual_svm(train_dataSet, train_labels, test_dataSet, test_labels,C)

		print '**********Question 4***************'
		for i in range(3):
			print '-------------------------------------------------------------------'
			kernel_svm.kernel_svm(train_dataSet, train_labels, test_dataSet, test_labels,i+1,Cat[i])

		print '**********Question 5***************'
		lamdas = [0.0001,0.001, 0.01, 0.1, 1, 2, 5, 6,10,100,1000]
		for lamda in lamdas:
			print '------------------------------------------------------------'
			print 'lamda = ', lamda
			logistic_regression.logistic_reg(train_dataSet, train_labels, test_dataSet, test_labels,lamda)


main(train_path, test_path)
