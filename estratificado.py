import pandas
dados = pandas.read_csv('breast_cancer.csv')
# tamanho da amostra estratificada
tamanho_amostra = 100
# obtendo as classes da base de dados
classes = dados['diagnosis'].unique()
# nesta lista armazenaremos, para cada classe, um
# pandas.DataFrame com suas amostras
amostras_por_classe = []
for c in classes:
    # obtendo os indices do DataFrame
    # cujas instancias pertencem a classe c
    indices_c = dados['diagnosis'] == c
    # extraindo do DataFrame original as observacoes da
    # classe c (obs_c sera um DataFrame tambem)
    obs_c = dados[indices_c]
    # calculando a proporcao de elementos da classe c
    # no DataFrame original
    proporcao_c = len(obs_c) / len(dados)
    # calculando a quantidade de elementos da classe
    # c que estarao contidos na amostra estratificada
    qtde_c = round(proporcao_c * tamanho_amostra)
    # extraindo a amostra da classe c
    # caso deseje-se realizar amostra com reposicao ou,
    # caso len(obs_c) < qtde_c, pode-se
    # informar o parametro replace=True
    amostra_c = obs_c.sample(qtde_c)
    # armazenando a amostra_c na lista de amostras
    amostras_por_classe.append(amostra_c)
    # concatenando as amostras de cada classe em
# um único DataFrame
amostra_estratificada = pd.concat(amostras_por_classe)