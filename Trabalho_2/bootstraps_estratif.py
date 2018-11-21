# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 19:22:47 2018

@author: cnorn
"""

import pandas as pd
#import numpy as np
import math

dataset = pd.read_csv('wine.data')
#dataset = pd.read_csv('ionosphere.data')
#dataset = pd.read_csv('breast-cancer-wisconsin.data')

#print(list(dataset.columns.values))

#print("Tamanho Dataset: ", len(dataset))
def bootstraps(dados, k_folds, col_target):
    # Nome/ID da coluna do dataset que contem as classes
    #col_target = "Class"
    
    #k_folds = 10
    
    classes = dataset[col_target].unique()
    print(classes)
    n_classes = len(classes)
    
    # Dividir datafame em novos dataframes de acordo com a classe ### 2 CLASSES
    if n_classes == 2:
        print("duas classes")
        dataset_a = dataset[dataset[col_target] == classes[0]]
        dataset_b = dataset[dataset[col_target] == classes[1]]
        
        dataset_a_size = len(dataset_a)
        dataset_b_size = len(dataset_b)
        
        bootstrap_a_size = math.ceil((dataset_a_size/k_folds))
        bootstrap_b_size = math.ceil((dataset_b_size/k_folds))
        
       
        bootstraps_a = []
        bootstraps_b = []
        
        bootstraps_a = [dataset_a[i:i+bootstrap_a_size] for i in range (0,dataset_a.shape[0],bootstrap_a_size)]
        bootstraps_b = [dataset_b[i:i+bootstrap_b_size] for i in range (0,dataset_b.shape[0],bootstrap_b_size)]
    
    
        #### Tratar erros na divisao em bootstraps quando divisão gera mais de K_folds####
        #print("bootA", len(bootstraps_a))
        if (len(bootstraps_a)) > k_folds:
            #print("tamanho bootstrap A", len(bootstraps_a))
            j = (len(bootstraps_a))-1
            #print("jota", j)
            
            while j > k_folds:
                #print("um", j)
                for i in range(len(bootstraps_a[j])):
                    if (len(bootstraps_a)) > k_folds:
                        #print("i" ,i)
                        bootstraps_a[i] = bootstraps_a[i].append((bootstraps_a[j].iloc[i]))
                        bootstraps_a[j] = (bootstraps_a[j]).drop(bootstraps_a[j].index[i])
                        if (len(bootstraps_a[j] == 0)):
                            print("del" , j)
                            del(bootstraps_a[j])
                        j = (len(bootstraps_a))-1
                    
                
                break
    
    
        if (len(bootstraps_b)) > k_folds:
            print("tamanho bootstrap b", len(bootstraps_a), k_folds)
            j = (len(bootstraps_b))-1
            print("jota", j)
            
            while j >= k_folds:
                print("um", j)
                for i in range(len(bootstraps_b[j])):
                    if (len(bootstraps_b)) > k_folds:
                        print("i" ,i, "j", j)
                        print("len j", len(bootstraps_b[j]))
                        bootstraps_b[i] = bootstraps_b[i].append((bootstraps_b[j].iloc[i]))
                        bootstraps_b[j] = (bootstraps_b[j]).drop(bootstraps_b[j].index[i])
                        print("len j 2", len(bootstraps_b[j]))
                        if len((bootstraps_b[j])) == 0:
                        #if (len(bootstraps_b[j] == 0)):    
                            print("del" , j)
                            del(bootstraps_b[j])
                            #print("lenA2", len(bootstraps_a), "k_folds", k_folds)
                        
                        j = (len(bootstraps_b))-1
                        #print("dois", j)
                    
                
                break
    
        
        #unir bootsraps/classes
        bootstraps_estratif = []
            
        for j in range(k_folds):
            bootstraps_estratif.append((bootstraps_a[j].append([bootstraps_b[j]])))
    
     
        #print(bootstraps_estratif[0])
    # Dividir datafame em novos dataframes de acordo com a classe # 3 CLASSES    
    if n_classes == 3:
        print("3 classes")
        
        dataset_a = dataset[dataset[col_target] == classes[0]]
        dataset_b = dataset[dataset[col_target] == classes[1]]
        dataset_c = dataset[dataset[col_target] == classes[2]]
        
        dataset_a_size = len(dataset_a)
        dataset_b_size = len(dataset_b)
        dataset_c_size = len(dataset_c)
        
        bootstrap_a_size = ((dataset_a_size//k_folds))
        bootstrap_b_size = ((dataset_b_size//k_folds))
        bootstrap_c_size = ((dataset_c_size//k_folds))
        
        bootstraps_a = []
        bootstraps_b = []
        bootstraps_c = []
        
        bootstraps_a = [dataset_a[a:a+bootstrap_a_size] for a in range (0,dataset_a.shape[0],bootstrap_a_size)]
        bootstraps_b = [dataset_b[b:b+bootstrap_b_size] for b in range (0,dataset_b.shape[0],bootstrap_b_size)]
        bootstraps_c = [dataset_c[c:c+bootstrap_c_size] for c in range (0,dataset_c.shape[0],bootstrap_c_size)]
        
        
        
        #### Tratar erros na divisao em bootstraps quando divisão gera mais de k_folds###
        if (len(bootstraps_a)) > k_folds:
            j = (len(bootstraps_a))-1
            
            while j > k_folds:
                for i in range(len(bootstraps_a[j])):
                    if (len(bootstraps_a)) > k_folds:
                        bootstraps_a[i] = bootstraps_a[i].append((bootstraps_a[j].iloc[i]))
                        bootstraps_a[j] = (bootstraps_a[j]).drop(bootstraps_a[j].index[i])
                        if (len(bootstraps_a[j] == 0)):
                            del(bootstraps_a[j])
                        j = (len(bootstraps_a))-1
                    
                
                break
    
    
        if (len(bootstraps_b)) > k_folds:
            j = (len(bootstraps_b))-1
            
            while j >= k_folds:
                for i in range(len(bootstraps_b[j])):
                    if (len(bootstraps_b)) > k_folds:
                        bootstraps_b[i] = bootstraps_b[i].append((bootstraps_b[j].iloc[i]))
                        bootstraps_b[j] = (bootstraps_b[j]).drop(bootstraps_b[j].index[i])
                        if len((bootstraps_b[j])) == 0:
                            del(bootstraps_b[j])
                        
                        j = (len(bootstraps_b))-1
                
                break
    
    
        if (len(bootstraps_c)) > k_folds:
            j = (len(bootstraps_c))-1
            
            while j > k_folds:
                for i in range(len(bootstraps_c[j])):
                    if (len(bootstraps_c)) > k_folds:
                        bootstraps_c[i] = bootstraps_c[i].append((bootstraps_c[j].iloc[i]))
                        bootstraps_c[j] = (bootstraps_c[j]).drop(bootstraps_c[j].index[i])
                        if (len(bootstraps_c[j] == 0)):
                            del(bootstraps_c[j])
                        
                        j = (len(bootstraps_c))-1
                    
                
                break
    
        
    
        #unir bootsraps/classes
        bootstraps_estratif = []
            
        for j in range(k_folds):
            bootstraps_estratif.append(bootstraps_a[j].append(([bootstraps_b[j].append(bootstraps_c[j])])))
           
    return bootstraps_estratif


teste = bootstraps(dataset, 10, "Class")

print(teste[0])
#print(teste[0])