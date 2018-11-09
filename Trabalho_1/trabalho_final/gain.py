from collections import Counter
import math
import numpy as np
from utils import *

def entropy_total(df):

    name_att_inf = df.keys()[-1]

    if len(df.shape) == 2:
        total = df.shape[0]
        cols = df.shape[1]-1
        labels = df.values[:, cols]
    else:
        total = 1
        labels = df.values[df.shape[0]-1]

    counters = Counter(','.join(df[name_att_inf]).replace(' ', '').split(','))
    entropia_total = 0
    if total != 1:
        for name_x in np.unique(labels):
            entropia_total +=-1*((counters.get(name_x)/total)*math.log((counters.get(name_x)/total), 2))
    return entropia_total


def ponto_corte_num(lista_dados, indice):
    soma = 0

    for i in range(0,len(lista_dados)):
            soma= soma + float(lista_dados[i][indice])

    ponto_corte = soma/len(lista_dados)
    return ponto_corte


def entropia_media_num(lista_dados, indice, col_target, ponto_corte):
    lista_classes = np.asarray(lista_dados)[:, col_target]
    lista_classes = list(set(lista_classes))

    total_menor = 0
    total_maior = 0

    count_menor_pos = np.zeros(len(lista_classes))
    count_maior_pos = np.zeros(len(lista_classes))

    for i in range(len(lista_dados)):
        if float(lista_dados[i][indice]) <= float(ponto_corte):
            for cme in range(len(lista_classes)):
                if lista_dados[i][col_target] == lista_classes[cme]:
                    # print("classe m:", lista_classes[cme])
                    count_menor_pos[cme]=count_menor_pos[cme]+1

            total_menor = total_menor + 1

        if float(lista_dados[i][indice]) > float(ponto_corte):
            for cma in range(len(lista_classes)):
                if lista_dados[i][col_target] == lista_classes[cma]:
                    # print("classe M", lista_classes[cma])
                    count_maior_pos[cma]=count_maior_pos[cma]+1

            total_maior=total_maior+1

    entropia_menor = 0.0
    entropia_maior = 0.0
    for i in range(len(lista_classes)):

        if count_menor_pos[i] != 0 and total_menor != 0:
            entropia_menor = entropia_menor + (-1*((count_menor_pos[i]/total_menor)*(math.log((count_menor_pos[i]/total_menor), 2))))

        if count_maior_pos[i] != 0 and total_maior != 0:

            entropia_maior = entropia_maior + (-1*((count_maior_pos[i]/total_maior)*math.log((count_maior_pos[i]/total_maior), 2)))

    count = len(lista_dados)

    ent_med_value = ((total_menor/count)*entropia_menor)+((total_maior/count)*entropia_maior)

    return ent_med_value


def most_gain(df):
    enable = True
    if enable:
        if df.shape[1] > 2:
            temp_df = df.values[:,df.shape[1]-1]
            temp_name = df.keys()[df.shape[1]-1]
            df = df.drop(columns=[df.keys()[df.shape[1]-1]])
            m_att = math.ceil(math.sqrt(df.shape[1]-1))
            df = df.sample(m_att, axis=1)
            df[temp_name] = temp_df


    entropy_total_value = entropy_total(df)

    if len(df.shape) == 2:
        total = df.shape[0]

    else:
        total = 1

    l_gain = []
    for column in range(0, df.shape[1]-1):

        labels = df.values[:, column]
        name_att_inf = df.keys()[column]
        dfa = df.set_index(name_att_inf, inplace=False)
        list_a = []
        ent_media = 0

        if isinstance((dfa.values[0, column]), float) or isinstance((dfa.values[0, column]), int):
            pto_corte = ponto_corte_num(dfa.values, column)
            col_target = dfa.shape[1] - 1
            ent_media = entropia_media_num(dfa.values, column, col_target, pto_corte)
        else:
            for a in np.unique(labels):
                list_a.append([a, entropy_total(dfa.loc[a]), len(dfa.loc[a])])

            for name, value, count in list_a:
                ent_media += count/total*value

        l_gain.append(entropy_total_value - ent_media)

    gain_final = l_gain[0]
    index = 0

    for i in range(0, len(l_gain)):
        if gain_final < l_gain[i]:
                index = i
                gain_final = l_gain[i]

    return index, gain_final
