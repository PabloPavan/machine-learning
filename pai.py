import numpy as np

def f(x):
	return x

def sig(x):
	return 1/(1+np.exp(-1*f(x)))

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
		print("Theta", c + 1, "inicial (pesos de cada neuronio, incluindo bias, armazenados nas linhas):\n", print2D(weights[c]))

	inputs=[]
	predictions=[]

	fdataset = open("dataset.txt", "r")
	i = 0
	for l in fdataset:
		a, b = l.split(";")

		inputs.append([])
		for v in a.split(","):
			inputs[i].append(float(v))

		predictions.append([])
		for v in b.split(","):
			predictions[i].append(float(v))

		i = i + 1
	
	print("Conjunto de treinamento")
	for l in range(0, len(inputs)):
		print("\tExemplo", l + 1)
		print("\t\tx:", print1D(inputs[l]))
		print("\t\ty:", print1D(predictions[l]))

	fdataset.close()

	print("\n--------------------------------------------")
	print("Calculando erro/custo J da rede")

	Jtotal = 0
	for example in range(0, len(inputs)):
		print("\tProcessando exemplo de treinamento", example + 1)
		print("\tPropagando entrada", print1D(inputs[example]))

		for layer in range(0, len(network) - 1):
			if layer == 0:
				input_propagate = inputs[example].copy()
				input_propagate.insert(0, 1)
			else:
				input_propagate = sig(z)
				input_propagate = np.insert(input_propagate, 0, 1)
			
			
			print("\t\ta" + str(layer + 1) + ":", print1D(input_propagate), "\n")

			z = np.dot(weights[layer], input_propagate)
			print("\t\tz" + str(layer + 2) + ":", print1D(z))

		input_propagate = sig(z)
		print("\t\ta" + str(len(network)) + ":", print1D(input_propagate))

		print("\n\t\tf(x):", print1D(input_propagate))

		print("\tSaida predita para o exemplo", example + 1, ":", print1D(input_propagate))
		print("\tSaida esperada para o exemplo", example + 1, ":", print1D(predictions[example]))

		J = -1*np.array(predictions[example]) * np.log(input_propagate) - (1 - np.array(predictions[example])) * np.log(1 - input_propagate)
		print("\tJ do exemplo", example + 1, ":", print1D(J), "\n")
		Jtotal += np.sum(J)

	Jtotal /= len(inputs)

	S = 0
	for layer in range(0, len(network) - 1):
		S += np.sum(np.delete(weights[layer], 0, axis=1) ** 2)
	S = regularization / (2 * len(inputs)) * S

	print("J total do dataset (com regularizacao): ", Jtotal + S)

	print("\n--------------------------------------------")
	print("Rodando backpropagation")
	for example in range(0, len(inputs)):
		print("\tCalculando gradientes com base no exemplo", example + 1)

if __name__ == "__main__":
    main() 