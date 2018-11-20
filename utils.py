import numpy as np
import logging
import logging.handlers

def func(x, func=True):
	if func:
		return 1/(1+np.exp(-1*f(x)))
	else:
		return f(x) * (f(x) > 0)

def f(x):
	return x

def normalize(data):
	return  data/data.max(axis=0)

def configureLog(logger):
		
		logger.setLevel(logging.DEBUG)

		
		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(processName)s - %(message)s')

		
		handler = logging.handlers.RotatingFileHandler("results.log", maxBytes=268435456, backupCount=50, encoding='utf8')

		handler.setFormatter(formatter)
		logger.addHandler(handler)

		logger.info('neural_network has started')

def major_voting(result_data, dict):
    count = result_data.count(dict[0])
    index = 0
    for i in range(0, len(dict)):
        if count <= result_data.count(dict[i]):
            index = i
    return dict[index]

def k_folds(dataset, i, k):
    n = len(dataset)
    data_test = dataset[n*(i-1)//k:n*i//k]
    data_train = dataset.drop(dataset.index[n*(i-1)//k:n*i//k])

    return data_train, data_test
