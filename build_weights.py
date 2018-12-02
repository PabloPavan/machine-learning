import numpy as np 
import argparse


def main():

	# parser = argparse.ArgumentParser(description='Argument')
	# parser.add_argument('--inputname', "--i",required=True,  metavar='FILE', type=str)
	# parser.add_argument('--outputname', "--o",required=True,  metavar='FILE', type=str)
	# args = parser.parse_args()
	

	network = np.loadtxt("net_io.txt", dtype='i', delimiter=',', skiprows=1)

	file = open("pesos_io.txt", 'w')

	for x in range(1, len(network)):  # linha por linha 
		final = []
		for y in range(0, network[x]): #valores para cada um dos neuronios 
			temp = []
			for z in range(0,network[x-1]+1): # +1 bias
				rand =  "%.2f" % float(np.random.sample(1))
				temp.append(str(rand))
			final.append(", ".join(temp))
		file.writelines("; ".join(final))
		file.writelines("\n")

	file.close()

if __name__ == "__main__":
    main()