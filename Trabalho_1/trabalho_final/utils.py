import pandas as pd
import numpy as np
import random


def k_folds(dataset, i, k):
    n = len(dataset)
    data_test = dataset[n*(i-1)//k:n*i//k]
    data_train = dataset.drop(dataset.index[n*(i-1)//k:n*i//k])

    return data_train, data_test


def major_voting(result_data, dict):
    count = result_data.count(dict[0])
    index = 0
    for i in range(0, len(dict)):
        if count <= result_data.count(dict[i]):
            index = i
    return dict[index]

def del_that_works(data, name):
    gambis_data = []

    for x in data:
        if (x != name):
            gambis_data.append(x)
    return gambis_data
