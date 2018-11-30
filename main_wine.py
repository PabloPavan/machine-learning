import numpy as np
from utils import *
from neural import *
from metrics import *
import math
from sklearn.utils import shuffle
import time
import sys
import csv

max_iterations  = 1000
epsilon         = 0.0000010000
num_kfolds      = 10

def main():

    inicio = time.time()


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

    fdataset = open("data/wine.data", "r")

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

    class1 = 0
    class2 = 0
    class3 = 0
    l_class = []

    class1_ver = np.array(([1],[0],[0]))
    class2_ver = np.array(([0],[1],[0]))
    class3_ver = np.array(([0],[0],[1]))

    
    for v_pred in predictions:
        if np.array_equal(class1_ver,v_pred):
            class1 += 1
        if np.array_equal(class2_ver,v_pred):
            class2 += 1
    class3 = len(predictions)-class1-class2
    l_class.append(class1)
    l_class.append(class2)
    l_class.append(class3)


    for valor in range(0, 3):
        l_class[valor] = int(l_class[valor]/num_kfolds)


    count_class1 = 0
    line_class1 = []
    count_class2 = 0
    line_class2 = []
    count_class3 = 0
    line_class3 = []


    for v_pred in predictions:
        if np.array_equal(class1_ver,v_pred):
            line_class1.append(count_class1)
        count_class1 += 1
        if np.array_equal(class2_ver,v_pred):
            line_class2.append(count_class2)
        count_class2 += 1
        if np.array_equal(class3_ver,v_pred):
            line_class3.append(count_class3)
        count_class3 += 1


    # print("1",line_class1)
    # print("2",line_class2)
    # print("3",line_class3)

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
        for k in range(0, l_class[2]):
            line = np.random.randint(0, len(line_class3))
            tempx.append(inputs[line_class3[line]])
            tempy.append(predictions[line_class3[line]])
            line_class3.remove(line_class3[line])
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
    for i in range(0, len(line_class3)):
        tempx.append(inputs[line_class3[i]])
        tempy.append(predictions[line_class3[i]])

    lista_kfold_input.append(tempx)
    lista_kfold_predition.append(tempy)

    k_f = 0
    f_mes = []
    
    while k_f != num_kfolds:
        
        weights=build_weights(network)
        
        train_input = []
        train_pred = []
        test_pred = lista_kfold_predition[k_f]
        test_input = lista_kfold_input[k_f]

        for w in range(0, num_kfolds):
            for j in range(0, len(lista_kfold_input[w])):
                if w != k_f:
                    train_input.append(lista_kfold_input[w][j])
                    train_pred.append(lista_kfold_predition[w][j])

        train_input, train_pred = shuffle(train_input, train_pred)
        test_input, test_pred = shuffle(test_input, test_pred)
        # here! add normalize data
        train_input = normalize(np.asarray(train_input))
        test_input = normalize(np.asarray(test_input))
        # print(print2D(test_input),print2D(test_pred))
        new_weigths = neural_network(network, weights, regularization, train_input, train_pred, max_iterations,alpha)
        k_f += 1

        results = feedfoward(network, new_weigths, test_input, test_pred)


        confusion_matrix = np.zeros(shape=(3, 3))

        for k in range(0, len(results)):
            for i in range(0, len(results[k])):
                if results[k][i] == 1:
                    break

            for j in range(0, len(test_pred[k])):
                if test_pred[k][j] == 1:
                    break

            confusion_matrix[i][j] += 1


        # for j in range(0, len(results)):

        #     if np.array_equal(class1_ver,results[j]) and np.array_equal(test_pred[j],results[j]):
        #         confusion_matrix[0][0] += 1
        #     elif np.array_equal(class2_ver,results[j]) and np.array_equal(test_pred[j],results[j]):
        #         confusion_matrix[1][1] += 1
        #     elif np.array_equal(class3_ver,results[j]) and np.array_equal(test_pred[j],results[j]):
        #         confusion_matrix[2][2] += 1
        #     elif not (np.array_equal(class1_ver,results[j])) and np.array_equal(test_pred[j],results[j]):
        #         if np.array_equal(class2_ver,results[j]):
        #             confusion_matrix[1][0] += 1
        #         elif np.array_equal(class3_ver,results[j]):
        #             confusion_matrix[2][0] += 1
        #     elif not (np.array_equal(class2_ver,results[j])) and np.array_equal(test_pred[j],results[j]):
        #         if np.array_equal(class1_ver,results[j]):
        #             confusion_matrix[0][1] += 1
        #         elif np.array_equal(class3_ver,results[j]):
        #             confusion_matrix[2][1] += 1
        #     elif not (np.array_equal(class3_ver,results[j])) and np.array_equal(test_pred[j],results[j]):
        #         if np.array_equal(class1_ver,results[j]):
        #             confusion_matrix[0][2] += 1
        #         elif np.array_equal(class2_ver,results[j]):
        #             confusion_matrix[1][2] += 1

        print(confusion_matrix)

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
    fim = time.time()

    print("lambda:", regularization)
    print("alpha:", alpha)
    print("media:", media_fmes)
    
    for q in range(0, len(f_mes)):
        variancia = math.pow(f_mes[q] - media_fmes, 2) / len(f_mes)
    print("variancia:", variancia)
    desvio_padrao = math.sqrt(variancia)
    print("desvio_padrao:", desvio_padrao)
    print("tempo:", fim - inicio)

if __name__ == "__main__":

    main()