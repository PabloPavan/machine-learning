import numpy as np
import random
def f(x):
	return x+np.sin(x)

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


