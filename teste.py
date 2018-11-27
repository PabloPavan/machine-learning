import numpy as np
from utils import *
from neural import *
from metrics import *
import math
from sklearn.utils import shuffle


weights = []
fweights = open("pesos_io.txt", "r")
i = 0
for l in fweights:
    weights.append([])
    j = 0
    for n in l.split(";"):
        weights[i].append([])
        k = 0
        for w in n.split(","):
            weights[i][j].append(float(w))
            k = k + 1
        j = j + 1
    weights[i] = np.array(weights[i])
    i = i + 1

fweights.close()

print(print3D(weights))

print(weights[2].shape)

lista = [] 

network = []

fnetwork = open("net_io.txt", "r")
regularization = float(fnetwork.readline())
for line in fnetwork:
    network.append(int(line))
fnetwork.close()

print(network)


finall = []
for x in range(1, len(network)):  # linha por linha 
        final = []
        for y in range(0, network[x]): #valores para cada um dos neuronios 
            temp = []
            for z in range(0,network[x-1]+1): # +1 bias
                rand =  "%.4f" % float(np.random.sample(1))
                temp.append(rand)
            final.append(temp)

        finall.append(np.asarray(final, dtype=float))

print(finall[2].shape)
print(print3D(finall))