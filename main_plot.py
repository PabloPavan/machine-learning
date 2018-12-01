import numpy as np
from utils import *
from neural_plot import *
import math
import sys
import csv

max_iterations  = 1000
epsilon         = 0.0000010000

def main():


    np.set_printoptions(precision=5)

    network = []

    fnetwork = open(sys.argv[1], "r")
    regularization = float(fnetwork.readline())
    alpha = float(fnetwork.readline())
    for line in fnetwork:
        network.append(int(line))
    fnetwork.close()

    inputs = []
    predictions = []

    fdataset = open(sys.argv[2], "r")

    i = 0
    for l in fdataset:
        a, b = l.split(";")
        inputs.append([])
        for v in a.split(","):
            inputs[i].append(float(v))

        predictions.append([])
        for v in b.split(","):
            predictions[i].append(float(v))

        i = i + 1

    for l in range(0, len(inputs)):
        inputs[l] = np.array(inputs[l], ndmin=2).T

    for l in range(0, len(predictions)):
        predictions[l] = np.array(predictions[l], ndmin=2).T
    fdataset.close()


    weights=build_weights(network)

    inputs = normalize(np.asarray(inputs))
        
    neural_network(network, weights, regularization, inputs, predictions, max_iterations,alpha, sys.argv[3])


if __name__ == "__main__":

    main()