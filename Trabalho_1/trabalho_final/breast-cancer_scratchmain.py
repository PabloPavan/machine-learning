import sys
import pandas as pd
import time
from bootstrap import *
from metrics import *
from decision_tree import *
from utils import *



def main():

    num_bootstraps = int(sys.argv[1])
    filename = 'breast-cancer-wisconsin.data'
    df = pd.read_csv(filename, sep=',', header=0)
    df = df.sample(n=df.__len__())
    df.drop('sample', axis='columns', inplace=True)

    root = []
    dict_inf_extracted = np.unique(df.values[:, -1])

    f_mes = []
    kfold = 10
    num = 1
    while num <= kfold :

        result = []
        result_gt = []
        result_voting = []
        data_train, data_test = k_folds(df, num, kfold)
        df_keys = list(data_train.keys())
        num_instances = data_train.shape[0]

        list_of_bootstraps_train = create_n_bootstraps_train(num_bootstraps, num_instances)

        list_bootstrap_train_list_data = bootstrap_n_data_list_train(num_bootstraps, num_instances, data_train,
                                                                     list_of_bootstraps_train)
        for w in range(0, num_bootstraps):
            d_train_ensemble = pd.DataFrame.from_records(list_bootstrap_train_list_data[w])
            d_train_ensemble.columns = df_keys
            root.append(decision_tree(l_data=d_train_ensemble, l_att=list(d_train_ensemble.keys())))

        for w in range(0, len(data_test)):
            instance = data_test.values[w][0:len(df_keys)-1]
            for j in range(0, len(root)):
                #print("instance:", instance)
                result.append(root[j].evaluate(instance, df_keys[:-1]))

            result_voting.append(major_voting(result, dict_inf_extracted))
            result_gt.append(data_test.values[w][-1])
        #montar confusion_matrix
        num_class = len(dict_inf_extracted)
        confusion_matrix = np.zeros(shape=(num_class, num_class))

        #muda pra cada teste
        class_dict = []

        #organiza o dicionario na ordem que queremos da matriz de confusao (manual)
        class_dict.append(dict_inf_extracted[0])
        class_dict.append(dict_inf_extracted[1])

        for j in range(0, len(result_voting)):

            if result_voting[j] == result_gt[j] and class_dict[0] == result_gt[j]:
                confusion_matrix[0][0] += 1

            if result_voting[j] == result_gt[j] and class_dict[1] == result_gt[j]:
                confusion_matrix[1][1] += 1

            if result_voting[j] != result_gt[j] and class_dict[0] == result_voting[j]:
                confusion_matrix[0][1] += 1

            if result_voting[j] != result_gt[j] and class_dict[1] == result_voting[j]:
                confusion_matrix[1][0] += 1
        num = num+1

        #print(confusion_matrix)
        prec_array = []
        rec_array = []
        #calcula prec e rec pra cada coluna

        for col in range(0, len(confusion_matrix)-1):
            conf_matrix_bin = confusion_matrix_bin(confusion_matrix, col)
            prec_array.append(precision(conf_matrix_bin))
            rec_array.append(recall(conf_matrix_bin))
        #media de prec e recal
        prec, rec = macro_median(prec_array, rec_array)
        beta = 1

        #calculo da f_measure para a macro median
        f1 = f_measure(prec, rec, beta)
        f_mes.append(f1)
        print(num_bootstraps, ";", num-1, ";", prec, ";", rec, ";", f1)

    media_fmes = sum(f_mes)/len(f_mes)
    print("media:", media_fmes)
    for q in range(0, len(f_mes)):
     variancia = math.pow(f_mes[q]-media_fmes, 2)/len(f_mes)
    print("variancia:", variancia)
    desvio_padrao = math.sqrt(variancia)
    print("desvio padrao", desvio_padrao)



if __name__ == "__main__":
   start_time = time.time()
   main()
   print("--- %s seconds ---" % (time.time() - start_time))

