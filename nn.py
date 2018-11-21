import numpy as np
import re
import math
import logging
import logging.handlers
from utils import *

class neural_network():

    def __init__(self, reg_value, network, initial_weights_file, dataset, logger):

        self.reg_value = reg_value
        self.network = network
        self.initial_weights_file = initial_weights_file
        self.dataset = dataset
        self.reg_cost = 0
        self.l_j_custo = []
        self.alfa = 0
        self.l_weight_layers = []
        self.l_activation_layers = []
        self.l_d_grad = []
        self.l_p_reg = []
        self.logger = logger


    def create_vectors(self): 

        for w_len in range(len(self.network)-1, 0, -1):
            self.d_grad = np.zeros((self.network[w_len-1]+1, self.network[w_len]), dtype=float)
            self.l_d_grad.append(self.d_grad)
            self.p_reg = np.zeros((self.network[w_len - 1] + 1, self.network[w_len]), dtype=float)
            self.l_p_reg.append(self.p_reg)

        # aqui criamos os arrays e preenchemos com os dados do arquivo txt

        for i in range(1, len(self.network)):

            self.l_weight_layers.append(np.zeros((self.network[i], self.network[i-1]+1), dtype=float))
            self.l_activation_layers.append(np.ones((self.network[i]+1, 1), dtype=float))
            line = self.initial_weights_file.readline()
            j = 0
            count = 0
            # need to put the values in the right place
            for value in re.findall(r'-?\d+\.?\d*', line):
                if count == (self.network[i-1])+1:
                    count = 0
                    j += 1
                self.l_weight_layers[i-1][j][count] = value
                count += 1

        #aqui preenchemos o array com a input
        
        # if "wine" in args.dataset:
        #     dataset_file = np.genfromtxt(,delimiter=',')
           

        #aqui posteriormente é feito um for para pegar todas as instancias do dataset e atualizar os pesos, etc
        #lê as instancias do dataset

    def training(self):
        itertion = 0
       
        while(itertion != 5000):

            self.num_lines = sum(1 for line in open(self.dataset))
            self.dataset_file = open(self.dataset, "r")
            
            for num_instance in range(0, self.num_lines):

                #print("exemplo:", num_instance+1)

                line = self.dataset_file.readline()
                self.instance_nn = np.zeros((self.network[0]+1, 1), dtype=float)
                list_values = []
                self.l_activation_layers_temp = []

                for value in re.findall(r'-?\d+\.?\d*', line):
                    list_values.append(value)
                self.logger.info('input propagation: {}'.format(list_values[0])) 
                

                self.instance_nn[0] = 1
                for j in range(0, self.network[0]):
                   self.instance_nn[j+1] = list_values[j]

                #print("a1", instance_nn)
                self.logger.info('a1: {}'.format(self.instance_nn)) 
                self.result_nn = np.zeros((self.network[len(self.network)-1], 1), dtype=float)
                pos = 0
                for j in range(0, self.network[len(self.network)-1]):
                    self.result_nn[pos] = list_values[self.network[0]+j]
                    pos += 1

                #atualizamos os valores dos pesos fazendo a multiplicacao pela matriz input
                #primeira vez temos que multiplicar pela input

                self.temp_result_layer = np.zeros((self.network[1], 1), dtype=float)
                np.matmul(self.l_weight_layers[0], self.instance_nn, self.temp_result_layer)
                self.logger.info('z2: {}'.format(self.temp_result_layer))
                
                self.temp_result_layer = func(self.temp_result_layer)
                self.l_activation_layers_temp.append(self.temp_result_layer)

                self.l_activation_layers[0][:] = 1
                for p in range(0, len(self.temp_result_layer)):
                    self.l_activation_layers[0][p+1] = self.temp_result_layer[p]
                self.logger.info('a2: {}'.format(self.l_activation_layers[0]))
                

                # a partir da segunda vez precisamos multiplicar pela activation layer anterior e
                #  adicionar uma linha com 1

                for w in range(1, len(self.network)-1):
                    self.temp_result_layer = np.zeros((self.network[w+1], 1), dtype=float)
                    np.matmul(self.l_weight_layers[w], self.l_activation_layers[w-1], self.temp_result_layer)

                    self.temp_result_layer = func(self.temp_result_layer)
                    self.l_activation_layers_temp.append(self.temp_result_layer)

                    
                    self.logger.info('a3: {}'.format(self.l_activation_layers_temp[w]))
                    self.result_inf = np.zeros((self.network[len(self.network)-1], 1), dtype=float)
                    self.result_inf = self.l_activation_layers_temp[len(self.l_activation_layers_temp)-1]
                    self.logger.info('z3: {}'.format(self.result_inf))
                    self.logger.info('expected result: {}'.format(self.result_nn))

                    self.backprogation()
                print('\r', "iteration=", itertion, end="", flush=True)

            itertion=itertion+1
                
            for layer in range(0, len(self.l_activation_layers_temp)-1):
                self.reg_cost = self.reg_cost+sum(self.l_activation_layers_temp[layer])**2
            self.reg_cost = self.reg_cost*self.reg_value/(2*self.num_lines)

        self.j_total = sum(self.l_j_custo)/self.num_lines+self.reg_cost


        

    def backprogation(self):

        for p in range(1, len(self.l_activation_layers_temp)):
            self.l_activation_layers[p][1:] = self.temp_result_layer

        # calcula os deltas
        # delta para camada de saída -> delta = f_theta(X)-y
        deltas = []
        delta_saida = np.zeros((self.network[len(self.network) - 1], 1), dtype=float)
        delta_saida = self.result_inf - self.result_nn
        deltas.append(delta_saida)

        # 1.3.Para cada camada k=L-1…2 // calcula os deltas para as camadas ocultas
        # 𝛿(l=k) = [θ(l=k)]T 𝛿(l=k+1) .* a(l=k) .* (1-a(l=k))
        count = 0
        for w_num in range(len(self.network)-2, 0, -1):
            delta_temp = np.ones((self.network[w_num], 1), dtype=float)
            delta_temp = np.transpose(self.l_weight_layers[w_num])*deltas[count]*self.l_activation_layers[w_num-1]*(1-self.l_activation_layers[w_num-1])
            count = count+1
            deltas.append(delta_temp[1:])

        ##calculo dos gradientes a fazer
        ##D(l=k) = D(l=k) + 𝛿(l=k+1) [a(l=k)]T
        count = len(self.l_weight_layers)-1
        for w_num in range(0, len(self.network)-1):

            if w_num != len(self.network)-2:
                self.l_d_grad[w_num] = self.l_d_grad[w_num] +deltas[w_num]*self.l_activation_layers[w_num]
            else:
                self.l_d_grad[w_num] = self.l_d_grad[w_num] +deltas[w_num]*np.transpose(self.instance_nn)

            self.l_p_reg[w_num] = self.reg_value*np.transpose(self.l_weight_layers[count])
            count = count - 1
        # print("self.l_d_grad")
        # print(self.l_d_grad)

        for w_num in range(len(self.network)-2, -1, -1):
            self.l_d_grad[w_num-1] = (1/self.num_lines)*(self.l_d_grad[w_num-1]+self.l_p_reg[w_num-1])

        for w_num in range(len(self.network) - 2, -1, -1):
            self.l_weight_layers[w_num] = self.l_weight_layers[w_num] - self.alfa*np.transpose(self.l_d_grad[w_num-1])

        count = len(self.l_d_grad)-1
        for w_num in range(0, len(self.network)-1):
            self.l_weight_layers[w_num] = self.l_weight_layers[w_num] - self.alfa*np.transpose(self.l_d_grad[count])
            count = count - 1
        # print(self.l_weight_layers)

        self.cost()

    def cost(self):
        #funcao de custo
        #aqui considerar log 10 nao 2!
        # na funcao de custo quando temos mais de uma saida o que fazemos?
        j_custo = -1*self.result_nn*(np.log(self.result_inf))-(1-self.result_nn)*(np.log(1-self.result_inf))
        #print("j_custo exemplo:", num_instance+1, j_custo)
        self.l_j_custo.append(j_custo)


## Backpropagation
