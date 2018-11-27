import numpy as np
from utils import *
from neural import *
from metrics import *
import math
from sklearn.utils import shuffle
#import argcomplete, argparse

max_iterations  = 1000
alpha           = 0.1
epsilon         = 0.0000010000
num_kfolds      = 10

def main():
    # parser = argparse.ArgumentParser(description='Argument')
    # parser.add_argument('--network', "--n",required=True,  metavar='FILE', type=str).completer = EnvironCompleter
    # parser.add_argument('--weights', "--w",required=True,  metavar='FILE', type=str).completer = EnvironCompleter
    # parser.add_argument('--dataset', "--d",required=True,  metavar='FILE', type=str).completer = EnvironCompleter
    # parser.add_argument('--log', "--l", required=False, default=True).completer = EnvironCompleter

    # argcomplete.autocomplete(parser)
    # args = parser.parse_args()

    np.set_printoptions(precision=5)

    network = []

    fnetwork = open("net_io.txt", "r")
    regularization = float(fnetwork.readline())
    for line in fnetwork:
        network.append(int(line))
    fnetwork.close()

    print("Parametro de regularizacao lambda=", regularization, "\n")
    print("Inicializando rede com a seguinte estrutura de neuronios por camadas:", network, "\n")

    # lista de matrizes de pesos
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

    # for c in range(0, len(weights)):
    #     print("Theta", c + 1, "inicial (pesos de cada neuronio, incluindo bias, armazenados nas linhas):\n",
    #           print2D(weights[c]))

    inputs = []
    predictions = []

    #fdataset = open("Trabalho_2/wine.data", "r")
    fdataset = open("data/ionosphere.data", "r")

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
    # print("Conjunto de treinamento")
    # for l in range(0, len(inputs)):
    #     print("\tExemplo", l + 1)
    #     print("\t\tx:", print1D(inputs[l]))
    #     print("\t\ty:", print1D(predictions[l]))
    #
    class1 = 0
    class2 = 0
    class3 = 0
    total_class = 2
    l_class = []

    if total_class == 2:
        for v_pred in predictions:
            if [[1, 0]] in v_pred.T:
                class1 += 1
        class2 = len(predictions)-class1
        l_class.append(class1)
        l_class.append(class2)
    # else:
    #     for v_pred in predictions:
    #        # print()
    #         if [[1, 0, 1]] in v_pred.T:
    #         #    print("a1")
    #             class1 += 1
    #         if [[0, 1]] in v_pred.T:
    #          #   print("a2")
    #             class2 += 1
        # print(len(predictions))
        # class3 = len(predictions) - class1 - class2


    for valor in range(0, total_class):
        l_class[valor] = int(l_class[valor]/num_kfolds)


    count_class1 = 0
    line_class1 = []
    count_class2 = 0
    line_class2 = []

    if total_class == 2:
        for v_pred in predictions:
            if [[1, 0]] in v_pred.T:
                line_class1.append(count_class1)
            count_class1 += 1
            if [[0, 1]] in v_pred.T:
                line_class2.append(count_class2)
            count_class2 += 1

    lista_kfold_input =[]
    lista_kfold_predition =[]

    for i in range(0, num_kfolds-1):
        tempx = []
        tempy = []
        for k in range(0, l_class[0]):
            line = np.random.randint(0, len(line_class1))
            tempx.append(inputs[line_class1[line]])
            tempy.append(predictions[line_class1[line]])
            line_class1.remove(line_class1[line])
        for k in range(0, l_class[1]):
            line = np.random.randint(0, len(line_class2))
            tempx.append(inputs[line_class2[line]])
            tempy.append(predictions[line_class2[line]])
            line_class2.remove(line_class2[line])
        lista_kfold_input.append(tempx)
        lista_kfold_predition.append(tempy)

    tempx = []
    tempy = []
    for i in range(0, len(line_class1)):
        tempx.append(inputs[line_class1[i]])
        tempy.append(predictions[line_class1[i]])
    for i in range(0, len(line_class2)):
        tempx.append(inputs[line_class2[i]])
        tempy.append(predictions[line_class2[i]])

    lista_kfold_input.append(tempx)
    lista_kfold_predition.append(tempy)



    # print("Conjunto de treinamento")
    # for l in range(0, len(lista_kfold_input)):
    #     print("\tExemplo", l + 1)
    #     print("\t\tx:", print1D(lista_kfold_input[0][l]))
    #     print("\t\ty:", print1D(lista_kfold_predition[0][l]))


    # build the test data
    k_f = 0
    f_mes = []
    while k_f != num_kfolds:
        train_input = []
        train_pred = []
        test_pred = lista_kfold_predition[k_f]
        test_input = lista_kfold_input[k_f]

        for w in range(0, num_kfolds):
            for j in range(0, len(lista_kfold_input[w])):
                if w != k_f:
                    train_input.append(lista_kfold_input[w][j])
                    train_pred.append(lista_kfold_predition[w][j])

        train_input, train_pred = shuffle(train_input, train_pred, random_state=0)
        # here! add normalize data
        #train_input = normalize(np.asarray(train_input))
        #test_input = normalize(np.asarray(test_input))

        new_weigths = neural_network(network, weights, regularization, train_input, train_pred, max_iterations,alpha)
        k_f += 1

        a = feedfoward(network, new_weigths, test_input, test_pred)

        class1 = np.array(([1],[0]))
        class2 = np.array(([0],[1]))


        # print(test_pred[0].shape)
        # print(a[0].shape)
        # print(tp.shape)

        confusion_matrix = np.zeros(shape=(2, 2))
        for j in range(0, len(a)):

            if np.array_equal(class1,a[j]) and np.array_equal(test_pred[j],a[j]):
                confusion_matrix[0][0] += 1
            elif np.array_equal(class2,a[j]) and np.array_equal(test_pred[j],a[j]):
                confusion_matrix[1][1] += 1
            elif np.array_equal(class1, a[j]) and not(np.array_equal(test_pred[j], a[j])):
                confusion_matrix[0][1] += 1
            elif np.array_equal(class2, a[j]) and not (np.array_equal(test_pred[j], a[j])):
                confusion_matrix[1][0] += 1

                # print(confusion_matrix)
        prec_array = []
        rec_array = []
        # calcula prec e rec pra cada coluna
        for col in range(0, len(confusion_matrix) - 1):
            conf_matrix_bin = confusion_matrix_bin(confusion_matrix, col)
            prec_array.append(precision(conf_matrix_bin))
            rec_array.append(recall(conf_matrix_bin))
        # media de prec e recal
        prec, rec = macro_median(prec_array, rec_array)
        beta = 1

        # calculo da f_measure para a macro median
        f1 = f_measure(prec, rec, beta)
        f_mes.append(f1)

    media_fmes = sum(f_mes)/len(f_mes)
    print("media:", media_fmes)
    for q in range(0, len(f_mes)):
        variancia = math.pow(f_mes[q] - media_fmes, 2) / len(f_mes)
    print("variancia:", variancia)
    desvio_padrao = math.sqrt(variancia)
    print("desvio padrao", desvio_padrao)

if __name__ == "__main__":

    main()