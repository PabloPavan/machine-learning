import numpy as np

def f(x):
	return x + x + x

def sig(x):
	return 1/(1+np.exp(-1*f(x)))

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
	return  data/data.max(axis=0)


def major_voting(result_data, dict):
    count = result_data.count(dict[0])
    index = 0
    for i in range(0, len(dict)):
        if count <= result_data.count(dict[i]):
            index = i
    return dict[index]

def k_folds(dataset, i, k):
    n = len(dataset)
    data_test = dataset[n*(i-1)//k:n*i//k]
    data_train = dataset.drop(dataset.index[n*(i-1)//k:n*i//k])

    return data_train, data_test
