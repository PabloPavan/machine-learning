import numpy as np
from utils import *
from neural_example import *
import argcomplete,argparse


def main():

    parser = argparse.ArgumentParser(description='Argument')
    parser.add_argument('--network', "--n",required=True,  metavar='FILE', type=str)
    parser.add_argument('--weights', "--w",required=True,  metavar='FILE', type=str)
    parser.add_argument('--dataset', "--d",required=True,  metavar='FILE', type=str)

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    
     
    np.set_printoptions(precision=5)
    
    network=[]

    fnetwork = open(args.network, "r")
    regularization = float(fnetwork.readline())
    for line in fnetwork:
        network.append(int(line))
    fnetwork.close()
    
    print("Parametro de regularizacao lambda=", regularization, "\n")
    print("Inicializando rede com a seguinte estrutura de neuronios por camadas:", network, "\n")

    # lista de matrizes de pesos
    weights=[]
    fweights = open(args.weights, "r")

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

    for c in range(0, len(weights)):
        print("Theta", c + 1, "inicial (pesos de cada neuronio, incluindo bias, armazenados nas linhas):\n", print2D(weights[c]))

    inputs=[]
    predictions=[]

    fdataset = open(args.dataset, "r")
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


    print("Conjunto de treinamento")
    for l in range(0, len(inputs)):
        print("\tExemplo", l + 1)
        print("\t\tx:", print1D(inputs[l]))
        print("\t\ty:", print1D(predictions[l]))

    fdataset.close()

    max_iterations  = 1
    alpha           = 0.001
    epsilon         = 0.0000010000

    neural_network(network,weights,regularization, inputs, predictions, max_iterations,alpha)


if __name__ == "__main__":

    main()