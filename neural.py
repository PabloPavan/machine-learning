import numpy as np
from utils import *


def neural_network(network, weights, regularization, inputs, predictions, max_iterations, alpha):
    iterations = 0
    while iterations < max_iterations:
        iterations = iterations + 1
        # print("\r Iteration", iterations)


        Jtotal = 0
        input_propagate = []
        for example in range(0, len(inputs)):
            input_propagate.append([])
            for layer in range(0, len(network) - 1):
                if layer == 0:
                    input_propagate[example].append(np.array(inputs[example], ndmin=2))
                else:
                    input_propagate[example].append(np.array(sig(z), ndmin=2))

                input_propagate[example][layer] = np.insert(input_propagate[example][layer], 0, 1, 0)

                z = np.dot(weights[layer], input_propagate[example][layer])

            layer = layer + 1
            input_propagate[example].append(np.array(sig(z), ndmin=2))

            J = -1 * predictions[example] * np.log(input_propagate[example][layer]) - (
                        1 - predictions[example]) * np.log(1 - input_propagate[example][layer])
            Jtotal += np.sum(J)

        Jtotal /= len(inputs)

        S = 0
        for layer in range(0, len(network) - 1):
            S += np.sum(np.delete(weights[layer], 0, axis=1) ** 2)
        S = regularization / (2 * len(inputs)) * S

        delta = []
        D = []
        for layer in range(0, len(network) - 1):
            D.append(np.zeros(weights[layer].shape))

        for example in range(0, len(inputs)):

            delta.append([])
            for layer in range(0, len(network)):
                delta[example].append([])

            delta[example][layer] = input_propagate[example][layer] - predictions[example]

            for layer in reversed(range(1, len(network) - 1)):
                delta[example][layer] = np.dot(weights[layer].T, delta[example][layer + 1]) * input_propagate[example][
                    layer] * (1 - input_propagate[example][layer])
                delta[example][layer] = np.delete(delta[example][layer], 0)
                delta[example][layer] = np.array(delta[example][layer], ndmin=2).T

            for layer in reversed(range(0, len(network) - 1)):
                Dtemp = np.dot(delta[example][layer + 1], input_propagate[example][layer].T)
                D[layer] = D[layer] + Dtemp

        P = []
        for layer in range(0, len(network) - 1):
            P.append([])

        for layer in range(0, len(network) - 1):
            weightsTemp = weights[layer].copy()
            weightsTemp[:, 0] = 0
            P[layer] = regularization * weightsTemp
            D[layer] = (1 / len(inputs)) * (D[layer] + P[layer])

        # print("\n--------------------------------------------")
        # print("Rodando verificacao numerica de gradientes (epsilon=" + str(epsilon) + ")")
        # for layer in range(0, len(network) - 1):
        # 	Dver = numVer(D[layer])
        # 	print("Gradientes finais para Theta" + str(layer + 1) + ":\n", print2D(Dver, tab="\t\t"))
        # 	print(print2D(delta[example]))

        for layer in range(0, len(network) - 1):
            weights[layer] = weights[layer] - alpha * D[layer]
    # 	print("Theta", layer + 1, "inicial (pesos de cada neuronio, incluindo bias, armazenados nas linhas):\n", print2D(weights[layer]))
    return weights


def feedfoward(network, weights, inputs, predictions):
    input_propagate = []
    for example in range(0, len(inputs)):

        input_propagate.append([])
        for layer in range(0, len(network) - 1):
            if layer == 0:
                input_propagate[example].append(np.array(inputs[example], ndmin=2))
            else:
                input_propagate[example].append(np.array(sig(z), ndmin=2))

            input_propagate[example][layer] = np.insert(input_propagate[example][layer], 0, 1, 0)

            z = np.dot(weights[layer], input_propagate[example][layer])

        layer = layer + 1
        input_propagate[example].append(np.array(sig(z), ndmin=2))

    print("\tSaida predita para o exemplo", example + 1, ":", print1D(input_propagate[example][layer]))
    print("\tSaida esperada para o exemplo", example + 1, ":", print1D(predictions[example]))

    return 0