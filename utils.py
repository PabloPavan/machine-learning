import numpy as np
import random
import math
def f(x):
	return x

def sig(x):
	return 1/(1+np.exp(-1*f(x)))

def tanh(x):
	return 2*sig(2*x)-1

def reLu(x):
	return f(x)*(f(x) > 0)

def numVer(A):
	return (f(A + epsilon) - f(A - epsilon)) / (2 * epsilon)

def print1Dmm(A):
	s = ""
	for i in A:
		s += ("%.5f " % i)
	return s

def print1Dm(A):
	s = ""
	for i in A:
		s = s + print1Dmm(i) + " "
	return s

def print1D(A):
	s = ""
	for i in A:
		s += ("%.5f " % i)
	return s

def print2D(A, tab="\t"):
	s = ""
	for i in A:
		s = s +  str(tab) + print1D(i) + "\n"
	return s

def print3D(A):
	s = ""
	for i in A:
		s = s + print2D(i) + "\n"
	return s

def normalize(data):

	return (data-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))


def build_weights(network):

	weights = []
	for x in range(1, len(network)):  # linha por linha 
	        temp1 = []
	        for y in range(0, network[x]): #valores para cada um dos neuronios 
	            temp2 = []
	            for z in range(0,network[x-1]+1): # +1 bias
	                #rand =  "%.4f" % float(np.random.sample(1))
	                r = 4*(math.sqrt(6/(network[0]+network[len(network)-1])))
	                rand = random.uniform(-r,r)


	                temp2.append(rand)
	            temp1.append(temp2)

	        weights.append(np.asarray(temp1, dtype=float))
	        
	return weights
