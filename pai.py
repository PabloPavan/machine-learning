import numpy as np

max_iterations 	= 1
alpha   		= 0.001
epsilon 		= 0.0000010000

def f(x):
	return x

def sig(x):
	return 1/(1+np.exp(-1*f(x)))

def numVer(A):
	return (f(A + epsilon) - f(A - epsilon)) / (2 * epsilon)

def print1Dmm(A):
	s = ""
	for i in A:
		s += ("%.5f " % i)
	return s

def print1Dm(A):
	s = ""
	for i in A:
		s = s + print1Dmm(i) + " "
	return s

def print1D(A):
	s = ""
	for i in A:
		s += ("%.5f " % i)
	return s

def print2D(A, tab="\t"):
	s = ""
	for i in A:
		s = s +  str(tab) + print1D(i) + "\n"
	return s

def print3D(A):
	s = ""
	for i in A:
		s = s + print2D(i) + "\n"
	return s


def main():
	np.set_printoptions(precision=5)
	
	network=[]

	fnetwork = open("network2.txt", "r")
	regularization = float(fnetwork.readline())
	for line in fnetwork:
		network.append(int(line))
	fnetwork.close()
	
	print("Parametro de regularizacao lambda=", regularization, "\n")
	print("Inicializando rede com a seguinte estrutura de neuronios por camadas:", network, "\n")

	# lista de matrizes de pesos
	weights=[]
	fweights = open("initial_weights2.txt", "r")

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
		weights[i] = np.array(weights[i])
		i = i + 1

	fweights.close()

	for c in range(0, len(weights)):
		print("Theta", c + 1, "inicial (pesos de cada neuronio, incluindo bias, armazenados nas linhas):\n", print2D(weights[c]))

	inputs=[]
	predictions=[]

	fdataset = open("dataset2.txt", "r")
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

	for l in range(0, len(inputs)):
		inputs[l] =	np.array(inputs[l], ndmin=2).T

	for l in range(0, len(predictions)):
		predictions[l] =	np.array(predictions[l], ndmin=2).T
	
	print("Conjunto de treinamento")
	for l in range(0, len(inputs)):
		print("\tExemplo", l + 1)
		print("\t\tx:", print1D(inputs[l]))
		print("\t\ty:", print1D(predictions[l]))

	fdataset.close()

	iterations = 0
	while iterations < max_iterations:
		iterations = iterations + 1
		print("\n--------------------------------------------")
		print("Iteration", iterations)
		print("\n--------------------------------------------")
		print("Calculando erro/custo J da rede")

		Jtotal = 0
		input_propagate = []
		for example in range(0, len(inputs)):
			print("\tProcessando exemplo de treinamento", example + 1)
			print("\tPropagando entrada", print1D(inputs[example]))
			input_propagate.append([])
			for layer in range(0, len(network) - 1):
				if layer == 0:
					input_propagate[example].append(np.array(inputs[example], ndmin=2))
				else:
					input_propagate[example].append(np.array(sig(z), ndmin=2))
				
				input_propagate[example][layer] = np.insert(input_propagate[example][layer], 0, 1, 0)

				print("\t\ta" + str(layer + 1) + ":", print1Dm(input_propagate[example][layer]), "\n")

				z = np.dot(weights[layer], input_propagate[example][layer])
				print("\t\tz" + str(layer + 2) + ":", print1D(z))
			layer = layer + 1
			input_propagate[example].append(np.array(sig(z), ndmin=2))
			print("\t\ta" + str(layer + 1) + ":", print1D(input_propagate[example][layer]))

			print("\n\t\tf(x):", print1D(input_propagate[example][layer]))

			print("\tSaida predita para o exemplo", example + 1, ":", print1D(input_propagate[example][layer]))
			print("\tSaida esperada para o exemplo", example + 1, ":", print1D(predictions[example]))

			J = -1*predictions[example] * np.log(input_propagate[example][layer]) - (1 - predictions[example]) * np.log(1 - input_propagate[example][layer])
			Jtotal += np.sum(J)
			print("\tJ do exemplo", example + 1, ":", np.sum(J), "\n")
		Jtotal /= len(inputs)

		S = 0
		for layer in range(0, len(network) - 1):
			S += np.sum(np.delete(weights[layer], 0, axis=1) ** 2)
		S = regularization / (2 * len(inputs)) * S

		print("J total do dataset (com regularizacao): ", Jtotal + S)

		print("\n--------------------------------------------")
		print("Rodando backpropagation")
		
		delta = []
		D = []
		for layer in range(0, len(network) - 1):
				D.append(np.zeros(weights[layer].shape))

		for example in range(0, len(inputs)):
			print("\tCalculando gradientes com base no exemplo", example + 1)

			delta.append([])
			for layer in range(0, len(network)):
				delta[example].append([])
			
			delta[example][layer] = input_propagate[example][layer] - predictions[example]
			print("\t\td" + str(layer + 1), print1Dm(delta[example][layer]))
			for layer in reversed(range(1, len(network) - 1)):
				delta[example][layer] = np.dot(weights[layer].T, delta[example][layer + 1]) * input_propagate[example][layer] * (1 - input_propagate[example][layer])
				delta[example][layer] = np.delete(delta[example][layer], 0)
				delta[example][layer] = np.array(delta[example][layer], ndmin=2).T
				print("\t\td" + str(layer + 1), print1Dm(delta[example][layer]))

			for layer in reversed(range(0, len(network) - 1)):
				print("\t\tGradientes de Theta" + str(layer + 1) + " com base no exemplo " + str(example + 1) + ":")
				Dtemp = np.dot(delta[example][layer + 1], input_propagate[example][layer].T)
				D[layer] = D[layer] + Dtemp
				print(print2D(Dtemp, "\t\t\t"))

		print("Dataset completo processado. Calculando gradientes regularizados")
		P = []
		for layer in range(0, len(network) - 1):
			P.append([])

		for layer in range(0, len(network) - 1):
			weightsTemp = weights[layer].copy()
			weightsTemp[:, 0] =  0
			P[layer] = regularization * weightsTemp
			D[layer] = (1 / len(inputs)) * (D[layer] + P[layer])
			print("Gradiente numerico de Theta" + str(layer + 1) + ":\n", print2D(D[layer], tab="\t\t"))
		
		# print("\n--------------------------------------------")
		# print("Rodando verificacao numerica de gradientes (epsilon=" + str(epsilon) + ")")
		# for layer in range(0, len(network) - 1):
		# 	Dver = numVer(D[layer])
		# 	print("Gradientes finais para Theta" + str(layer + 1) + ":\n", print2D(Dver, tab="\t\t"))
			# print(print2D(delta[example]))

		for layer in range(0, len(network) - 1):
			weights[layer] = weights[layer] - alpha * D[layer]
		# 	print("Theta", layer + 1, "inicial (pesos de cada neuronio, incluindo bias, armazenados nas linhas):\n", print2D(weights[layer]))



if __name__ == "__main__":
    main() 