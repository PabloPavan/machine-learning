import numpy as np
import re
import math
import copy
import pprint as p

def printVector(A):
    for i in A:
        print("%.5f" % i, end=" ")
    print()

def printArray(A):
    for i in A:
        print(end="\t")
        for j in i:
            print("%.5f" % j, end=" ")
        print(end="\n")
    print()

def main():

    reg_file = np.loadtxt("network.txt", dtype='float', delimiter=',', skiprows=0)
    reg_value = reg_file[0]
    print("Parametro de regularizacao lambda=%.3f\n" %  reg_value)
    reg_cost = 0
    network = np.loadtxt("network.txt", dtype='i', delimiter=',', skiprows=1)
    print("Inicializando rede com a seguinte estrutura de neuronios por camadas: ", network, "\n")
    #read initial weights

    initial_weights_file = open("initial_weights.txt", "r")
#     l_j_custo = []
#     alfa = 0
    l_weight_layers = []
    l_activation_layers = []

    l_d_grad = []
    l_p_reg = []
    for w_len in range(len(network)-1, 0, -1):
        d_grad = np.zeros((network[w_len], network[w_len-1]+1), dtype=float)
        l_d_grad.append(d_grad)
        p_reg = np.zeros((network[w_len], network[w_len-1]+1), dtype=float)
        l_p_reg.append(p_reg)

#     # aqui criamos os arrays e preenchemos com os dados do arquivo txt

    for i in range(1, len(network)-1):
        l_activation_layers.append(np.ones((network[i]+1, 1), dtype=float))

    # print(l_activation_layers)
    for i in range(1, len(network)):
        l_weight_layers.append(np.zeros((network[i], network[i-1]+1), dtype=float))
        line = initial_weights_file.readline()
        j = 0
        count = 0
        # need to put the values in the right place
        for value in re.findall(r'-?\d+\.?\d*', line):
            if count == (network[i-1])+1:
                count = 0
                j += 1
            l_weight_layers[i-1][j][count] = value
            count += 1

    print("Theta1 inicial (pesos de cada neuronio, incluindo bias, armazenados nas linhas):")
    printArray(l_weight_layers[0])
#     #aqui preenchemos o array com a input
    num_lines = sum(1 for line in open('dataset.txt'))

    dataset_file = open("dataset.txt", "r")


#     #aqui posteriormente é feito um for para pegar todas as instancias do dataset e atualizar os pesos, etc
#     #lê as instancias do dataset
    # for num_instance in range(0, num_lines):
    for num_instance in range(0, 1):

        print("exemplo:", num_instance+1)
        line = dataset_file.readline()
        instance_nn = np.zeros((network[0]+1, 1), dtype=float)
        list_values = []
        l_activation_layers_temp = []

        for value in re.findall(r'-?\d+\.?\d*', line):
            list_values.append(value)

        print("\nPropagando entrada ", end="")
        for l in list_values:
            print(l, end=" ")
        print()

        instance_nn[0] = 1
        for j in range(0, network[0]):
           instance_nn[j+1] = list_values[j]
        # instance_nn é a input com o bias!
        print("a1")
        printArray(instance_nn)
        result_nn = np.zeros((network[len(network)-1], 1), dtype=float)

        #resultado esperado!
        for j in range(0, network[len(network)-1]):
            result_nn[j] = list_values[network[0]+j]

#         #atualizamos os valores dos pesos fazendo a multiplicacao pela matriz input slide 126
#         #primeira vez temos que multiplicar pela input
        # temp_result_layer = np.zeros((network[1], 1), dtype=float)
        temp_result_layer = np.dot(l_weight_layers[0], instance_nn)
        print("w1")
        printArray(l_weight_layers[0])

        print("z2", temp_result_layer)
        printArray(temp_result_layer)

#         # sigmoide! AQUI ADICIONAR  A HIPOTESE!!!
#         #l_activation_layer_temp SEM O BIAS!
        temp_result_layer = 1/(1+np.exp(-1*temp_result_layer))
        l_activation_layers_temp.append(temp_result_layer)
        print("z2 sign", temp_result_layer)

        l_activation_layers[0][:] = 1
        for p in range(0, len(temp_result_layer)):
            l_activation_layers[0][p+1] = temp_result_layer[p]
        print("a2", l_activation_layers[0])

#         # a partir da segunda vez precisamos multiplicar pela activation layer anterior e
#         #  adicionar uma linha com 1

        for w in range(1, len(network)-1):
            # temp_result_layer = np.zeros((network[w+1], 1), dtype=float)
            temp_result_layer = np.dot(l_weight_layers[w], l_activation_layers[w-1])
#             np.matmul(l_weight_layers[w], l_activation_layers[w-1], temp_result_layer)

#             # sigmoide! AQUI ADICIONAR  A HIPOTESE!!!
#             #print("z", w+2, temp_result_layer)
#             temp_result_layer = 1/(1+np.exp(-1*temp_result_layer))
#             l_activation_layers_temp.append(temp_result_layer)
#             result_inf = np.zeros((network[len(network) - 1], 1), dtype=float)
#             if w == len(network)-2:
#                 result_inf = l_activation_layers_temp[len(temp_result_layer)]
#              #   print("a", w+2, result_inf)
#              #   print("saida esperada", result_nn)
#              #   print("resultado predito:", result_inf)

#             else:
#                 l_activation_layers[w][:] = 1
#                 for p in range(0, len(temp_result_layer)):
#                     l_activation_layers[w][p + 1] = temp_result_layer[p]
#             #    print("a", w+2, l_activation_layers[w])

#     #         #BACKPROPAGATION!


#             # calcula os deltas
#             # delta para camada de saída -> delta = f_theta(X)-y
#             deltas = []
#             delta_saida = np.zeros((network[len(network) - 1], 1), dtype=float)
#             delta_saida = result_inf - result_nn
#             deltas.append(delta_saida)
#             #print("deltas:", delta_saida)
#             # 1.3.Para cada camada k=L-1…2 // calcula os deltas para as camadas ocultas
#             # 𝛿(l=k) = [θ(l=k)]T 𝛿(l=k+1) .* a(l=k) .* (1-a(l=k))
#             count = 0
#            # print(l_weight_layers)
#            # print(deltas)
#             for w_num in range(len(network)-2, 0, -1):
#                 delta_temp = np.matmul(np.transpose(l_weight_layers[w_num]), deltas[count])
#                 delta_temp = delta_temp*l_activation_layers[w_num-1]*(1-l_activation_layers[w_num-1])
#                 count = count+1
#                 deltas.append(delta_temp[1:])
#             print("deltas:")
#             print(deltas)

#             #calculo dos gradientes a fazer
#             #D(l=k) = D(l=k) + 𝛿(l=k+1) [a(l=k)]T
#             w_num = len(deltas)-1

#             for count in range(len(deltas)-2, -1, -1):

#                 print("asoisaj")
#                 print(d_grad_temp)
#                 l_d_grad[count] = d_grad_temp[0] + deltas[count]*np.transpose(l_activation_layers[count-1])
#                 w_num = w_num - 1
#                 #l_d_grad_final.append(d_grad_temp)


#             print(l_d_grad)
#             # w_num = len(deltas)-1
#             # d_grad_temp = np.zeros((network[w_num], network[w_num-1] + 1), dtype=float)
#             # l_d_grad[w_num] = l_d_grad[w_num] + deltas[w_num]*np.transpose(instance_nn)
#             # l_d_grad_final.append(d_grad_temp)

#             #print(l_d_grad_final)
#             #l_d_grad = copy.deepcopy(l_d_grad_final)
#             # w_num = len(network)-1
#             # l_d_grad[w_num] = l_d_grad[len(network)] + deltas[count]*np.transpose(instance_nn)
#             # l_p_reg[w_num] = reg_value*np.transpose(l_weight_layers[w-1])
#             # print(l_d_grad[w_num])


#             # for w_num in range(len(network)-2, -1, -1):
#             #     l_d_grad[w_num-1] = (1/num_lines)*(l_d_grad[w_num-1]+l_p_reg[w_num-1])
#             #
#             # for w_num in range(len(network) - 2, -1, -1):
#             #     l_weight_layers[w_num] = l_weight_layers[w_num] - alfa*np.transpose(l_d_grad[w_num-1])
#             #
#             # count = len(l_d_grad)-1
#             # for w_num in range(0, len(network)-1):
#             #     l_weight_layers[w_num] = l_weight_layers[w_num] - alfa*np.transpose(l_d_grad[count])
#             #     count = count - 1
#             # print(l_weight_layers)
#     #
#     # #
#     # #
#     # #         #funcao de custo
#     # #
#     # #         ##aqui considerar log 10 nao 2!
#     # #         # na funcao de custo quando temos mais de uma saida o que fazemos?
#     #         j_custo = -1*result_nn*(np.log(result_inf))-(1-result_nn)*(np.log(1-result_inf))
#     #         print("j_custo exemplo:", num_instance+1, j_custo)
#     #         l_j_custo.append(j_custo)
#     #
#     #     #J+S!!
#     #     for layer in range(0, len(l_activation_layers_temp)-1):
#     #         reg_cost = reg_cost+sum(l_activation_layers_temp[layer])**2
#     #     reg_cost = reg_cost*reg_value/(2*num_lines)
#     #
#     # j_total = sum(l_j_custo)/num_lines+reg_cost
#     # print("j total", j_total)

#     ### Backpropagation


if __name__ == "__main__":
    main()