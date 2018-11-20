import numpy as np

dataset_file = np.genfromtxt('data/wine.data',delimiter=',', skip_header=1)
dataset_file = np.delete(dataset_file, obj=13, axis=1)

dataset_file = np.genfromtxt('data/ionosphere.data',delimiter=',', skip_header=1)
dataset_file = np.delete(dataset_file, obj=34, axis=1)
dataset_file = np.delete(dataset_file, obj=1, axis=1)

# dataset_file = np.genfromtxt('data/breast-cancer-wisconsin.data',delimiter=',', skip_header=1)
# dataset_file = np.delete(dataset_file, obj=10, axis=1)


print(dataset_file)

 x_normed = dataset_file / dataset_file.max(axis=0)

 print(x_normed)