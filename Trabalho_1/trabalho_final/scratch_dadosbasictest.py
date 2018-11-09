import pandas as pd
from decision_tree import *
from metrics import *
from gain import *
from utils import *


def main():

    filename = 'dadosBenchmark_validacaoAlgoritmoAD.csv'
    df = pd.read_csv(filename, sep=';', header=0)

    root = []

    instance = []
    result_voting = []
    result_gt = []
    list_att  = ["Tempo", "Temperatura", "Umidade", "Ventoso"]
    instance.append(["Chuvoso", "Quente", "Alta", "Verdadeiro"])
    result_gt.append("Nao")

    instance.append(["Ensolarado", "Quente", "Normal", "Falso"])
    result_gt.append("Sim")

    instance.append(["Nublado", "Quente", "Alta", "Falso"])
    result_gt.append("Sim")

    instance.append(["Chuvoso", "Quente", "Alta", "Falso"])
    result_gt.append("Sim")

    instance.append(["Chuvoso", "Quente", "Alta", "Verdadeiro"])
    result_gt.append("Nao")


    root = decision_tree(l_data=df, l_att=list(df.keys()))
    print(root)

    for i in range(0, len(instance)):
        result_voting.append(root.evaluate(instance[i], list_att))
    print(result_voting)
    print(result_gt)



if __name__ == "__main__":
    main()


