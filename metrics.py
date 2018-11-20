import numpy as np
from utils import *


def precision(confusion_matrix):
    if confusion_matrix[0][0] == 0:
        return 0
    else:
        prec = confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][0])
    return prec


def recall(confusion_matrix):
    if confusion_matrix[0][0] == 0:
        return 0
    else:
        rev = confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[0][1])
    return rev


def f_measure(prec, rev, beta):
    if prec == 0 and rev == 0:
        return 0
    else:
        f1_mes = (1+beta**2)*(prec*rev)/((beta**2)*prec+rev)
    return f1_mes


def macro_median(prec_array, rev_array):
        prec_med = sum(prec_array)/len(prec_array)
        rev_med = sum(rev_array)/len(rev_array)
        return prec_med, rev_med


def confusion_matrix_bin(confusion_matrix, col):

    confusion_matrix_binary = np.zeros(shape=(2, 2))
    #VP
    confusion_matrix_binary[0][0] = confusion_matrix[col][col]
    #FN
    confusion_matrix_binary[0][1] = (confusion_matrix.sum(axis=1))[col] - confusion_matrix[col][col]
    #FP
    confusion_matrix_binary[1][0] = confusion_matrix.sum(axis=0)[col] - confusion_matrix[col][col]
    #VN
    confusion_matrix_binary[1][1] = sum(confusion_matrix.sum(axis=1)) - confusion_matrix_binary[0][0] - confusion_matrix_binary[0][1] - confusion_matrix_binary[1][0]
    return confusion_matrix_binary

