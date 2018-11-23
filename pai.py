import numpy as np

def print1D(A):
	s = ""
	for i in A:
		s += ("%.5f " % i)
	return s

def print2D(A):
	s = ""
	for i in A:
		s = s +  "\t" + print1D(i) + "\n"
	return s

def print3D(A):
	s = ""
	for i in A:
		s = s + print2D(i) + "\n"
	return s


def main():
	np.set_printoptions(precision=5)
	
	network=[]

	fnetwork = open("network.txt", "r")
	regularization = float(fnetwork.readline())
	for line in fnetwork:
		network.append(int(line))
	fnetwork.close()
	
	print("Parametro de regularizacao lambda=", regularization, "\n")
	print("Inicializando rede com a seguinte estrutura de neuronios por camadas:", network, "\n")

	# lista de matrizes de pesos
	weights=[]

	fweights = open("initial_weights.txt", "r")

	i = 0
	for l in fweights:
		weights.append([])
		j = 0
		for n in l.split(";"):
			weights[i].append([])
			k = 0
			for w in n.split(","):
				weights[i][j].append(float(w))
				k = k + 1
			j = j + 1
		i = i + 1
	
	fweights.close()

	for c in range(0, len(weights)):
		print("Theta", c, "inicial (pesos de cada neuronio, incluindo bias, armazenados nas linhas):\n", print2D(weights[c]))

	fdataset = open("dataset.txt", "r")
	
	fdataset.close()



if __name__ == "__main__":
    main() 