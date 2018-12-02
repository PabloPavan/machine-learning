import numpy as np
from utils import *
# import matplotlib
# import matplotlib.pyplot as plt

epsilon = 0.00001
def neural_network(network, weights, regularization, inputs, predictions, max_iterations, alpha, name_fig):
    iterations = 0
    Jtotal = 0
    while iterations < max_iterations:
        iterations = iterations+1
#        print("\r Iteration", iterations)

        input_propagate = []
        prev_J = Jtotal 
        Jtotal = 0
        for example in range(0, len(inputs)):
            input_propagate.append([])
            for layer in range(0, len(network)-1):
                if layer == 0:
                    input_propagate[example].append(np.array(inputs[example], ndmin=2))
                else:
                    input_propagate[example].append(np.array(sig(z), ndmin=2))

                input_propagate[example][layer] = np.insert(input_propagate[example][layer], 0, 1, 0)

                z = np.dot(weights[layer], input_propagate[example][layer])

            layer = layer + 1
            input_propagate[example].append(np.array(sig(z), ndmin=2))

            # -np.average(y * np.log(p) + (1 - y) * np.log(1 - p))
            # J = log_loss + regTerm * np.linalg.norm(theta[1:]) / (2 * m)

            J = -1*predictions[example]*np.log(input_propagate[example][layer]+epsilon)-(1-predictions[example])*np.log(1-input_propagate[example][layer]+epsilon)
            Jtotal += np.sum(J)

        Jtotal/=len(inputs)
        
        S = 0
        for layer in range(0, len(network)-1):
            S += np.sum(np.delete(weights[layer], 0, axis=1)**2)
        S = regularization/(2*len(inputs))*S
        

        # plt.scatter(iterations, Jtotal+S, c="g")
        # plt.pause(0.001)
        # plt.draw()
        # plt.pause(0.05)
        # Jtotal = Jtotal+S 
            # print("iterations:", iterations)
            # print("J", Jtotal)
            # print("Prev J", prev_J)
            # print("Error", abs(prev_J-Jtotal))

        delta = []
        D = []
        for layer in range(0, len(network)-1):
            D.append(np.zeros(weights[layer].shape))

        for example in range(0, len(inputs)):

            delta.append([])
            for layer in range(0, len(network)):
                delta[example].append([])

            delta[example][layer] = input_propagate[example][layer]-predictions[example]

            for layer in reversed(range(1, len(network)-1)):
                delta[example][layer] = np.dot(weights[layer].T,delta[example][layer + 1])*input_propagate[example][layer]*(1 - input_propagate[example][layer])
                delta[example][layer] = np.delete(delta[example][layer], 0)
                delta[example][layer] = np.array(delta[example][layer], ndmin=2).T

            for layer in reversed(range(0, len(network)-1)):
                Dtemp = np.dot(delta[example][layer+1], input_propagate[example][layer].T)
                D[layer] = D[layer]+Dtemp

        P = []
        for layer in range(0, len(network)-1):
            P.append([])

        for layer in range(0, len(network)-1):
            weightsTemp = weights[layer].copy()
            weightsTemp[:, 0] = 0
            P[layer] = regularization*weightsTemp
            D[layer] = (1/len(inputs))*(D[layer]+P[layer])


        for layer in range(0,len(network)-1):
            weights[layer]=weights[layer]-alpha*D[layer]


        if iterations > 1 and abs(prev_J-Jtotal) <= 0.0001:
            max_iterations = iterations

    # plt.xlabel('EPHOCS')
    # plt.ylabel('J Total + Regularização')
    # plt.savefig(name_fig)
    return weights


def feedfoward(network, weights, inputs, predictions):
    input_propagate = []
    output = []
    for example in range(0, len(inputs)):
        input_propagate.append([])
        for layer in range(0, len(network)-1):
            if layer == 0:
                input_propagate[example].append(np.array(inputs[example], ndmin=2))
            else:
                input_propagate[example].append(np.array(sig(z), ndmin=2))

            input_propagate[example][layer] = np.insert(input_propagate[example][layer], 0, 1, 0)

            z = np.dot(weights[layer], input_propagate[example][layer])

        layer = layer + 1
        input_propagate[example].append(np.array(sig(z), ndmin=2))

        #input_propagate[example][layer] = np.around(input_propagate[example][layer])
        if len(input_propagate[example][layer]) == 2:
            if input_propagate[example][layer][0] >= input_propagate[example][layer][1]:
                input_propagate[example][layer][0] = 1 
                input_propagate[example][layer][1] = 0
            else:
                input_propagate[example][layer][0] = 0
                input_propagate[example][layer][1] = 1
        else:
            if input_propagate[example][layer][0] >= input_propagate[example][layer][1] and (input_propagate[example][layer][0] >= input_propagate[example][layer][2]):
                input_propagate[example][layer][0] = 1 
                input_propagate[example][layer][1] = 0
                input_propagate[example][layer][2] = 0
            elif input_propagate[example][layer][1] >= input_propagate[example][layer][0] and (input_propagate[example][layer][1] >= input_propagate[example][layer][2]):
                input_propagate[example][layer][0] = 0
                input_propagate[example][layer][1] = 1
                input_propagate[example][layer][2] = 0
            else:
                input_propagate[example][layer][0] = 0
                input_propagate[example][layer][1] = 0
                input_propagate[example][layer][2] = 1

#         print("\tSaida predita para o exemplo :", print1D(input_propagate[example][layer]))
#         print("\tSaida esperada para o exemplo :", print1D(predictions[example]))
        output.append(input_propagate[example][layer])
    return output
