import random
from utils import *


def create_bootstrap_train(num_instances, seed):

    bootstrap_train = []
    #random.seed(seed)

    for i in range(0, num_instances):
        bootstrap_train.append(random.randint(1, num_instances-1))

    return bootstrap_train


def create_n_bootstraps_train(num_bootstraps, num_instances):
    list_of_bootstraps_train = []
    seed = [1522, 5092, 1706, 930, 6598, 7823, 5078, 2884, 7383, 1519, 4482, 4851, 8336, 2684, 3116, 8221, 4631, 1926, 2451, 8958, 144, 5629, 9477, 7556, 7657, 8081, 382, 769, 6867, 9298, 4124, 2998, 2709, 5818, 9812, 8526, 7194, 4641, 8273, 4323, 9255, 9520, 1329, 954, 6797, 1920, 5412, 3295, 9821, 8805, 7779, 9260, 3037, 6826, 1332, 1515, 6007, 5727, 3795, 1557, 8707, 3387, 8857, 3903, 4863, 8549, 6363, 5578, 7425, 1967, 982, 3892, 4635, 753, 8224, 2694, 6702, 9039, 8540, 4601, 8730, 7645, 7258, 7467, 8598, 3335, 7934, 8606, 8852, 2640, 8602, 4479, 7402, 1645, 9443, 9014, 3315, 8494, 506, 2811]
    for i in range(0, num_bootstraps):

        bootstrap_train = create_bootstrap_train(num_instances, seed[i])
        list_of_bootstraps_train.append(bootstrap_train)

    return list_of_bootstraps_train


def bootstrap_data_list_train(num_instances, actual_bootstrap, list_data, list_of_bootstraps_train):
    bootstrap_list_data = []
    #bootstrap_list_data.append(list_data.keys())
    for i in range(0, num_instances):
        #print(actual_bootstrap, i, list_of_bootstraps_train[actual_bootstrap][i])
        #print( lista_dados[list_of_bootstraps_train[actual_bootstrap][i] ])
        bootstrap_list_data.append(list_data.values[list_of_bootstraps_train[actual_bootstrap][i]])

    return bootstrap_list_data


def bootstrap_n_data_list_train(numbootstraps, numinstances, lista_dados, list_of_bootstraps_train):
    list_bootstrap__list_data = []

    for actual_bootstrap in range(0, numbootstraps):
        #print(actual_bootstrap)
        bootstrap_train_list_data = bootstrap_data_list_train(numinstances, actual_bootstrap, lista_dados, list_of_bootstraps_train, )
        list_bootstrap__list_data.append(bootstrap_train_list_data)

    return list_bootstrap__list_data


def create_n_bootstraps_teste(num_instances, list_of_bootstraps_train):
    list_of_bootstraps_teste = []

    for i in range(0, len(list_of_bootstraps_train)):
        bootstrap_teste = create_bootstrap_teste(num_instances, list_of_bootstraps_train[i])
        list_of_bootstraps_teste.append(bootstrap_teste)

    return list_of_bootstraps_teste


def create_bootstrap_teste(numinstances, bootstrap_train):

    bootstrap_teste = []
    for w in range(0, numinstances):
        if bootstrap_train.count(w) == 0:
            bootstrap_teste.append(w)


    return bootstrap_teste


def bootstrap_data_list_teste (actual_bootstrap_teste, list_of_bootstraps_teste, list_data):
    bootstrap_list_teste_data = []
    tam = len(list_of_bootstraps_teste[actual_bootstrap_teste])
    bootstrap_list_teste_data.append(list_data.keys())
    for i in range(0, tam):
        #print(0, i, list_of_bootstraps_teste[actual_bootstrap_teste][i])
        #print(lista_dados[list_of_bootstraps_teste[actual_bootstrap_teste][i]])
        bootstrap_list_teste_data.append(list_data.values[list_of_bootstraps_teste[actual_bootstrap_teste][i]])

    return bootstrap_list_teste_data


def bootstrap_n_data_list_teste (numbootstraps, list_of_bootstraps_teste, lista_dados):
    list_bootstrap_list_teste_data = []

    for actual_bootstrap in range(0, numbootstraps):
        bootstrap_train_list_data = bootstrap_data_list_teste(actual_bootstrap, list_of_bootstraps_teste, lista_dados)
        list_bootstrap_list_teste_data.append(bootstrap_train_list_data)

    return list_bootstrap_list_teste_data