#!/bin/bash

# lambda: 0.01
# rede: [33, 4, 2]
# alpha: 0.01
# media: 0.313624338624
# variancia: 0.019042371784104582
# desvio_padrao: 0.13799410054094552
# tempo: 0.21584486961364746

# echo "network,lambda,alpha,f1_mean,variance,standard_deviation,time_execution" > $HOST"_"$DATE"_"ionosphere.csv

for lambda in 0 0.001; do
	for alpha in 0.01 0.1; do
		for layers in 1 4; do
	 		for neuron in 2 16; do 
	 	    	sbatch breast.batch $lambda $alpha $layers $neuron
	 		done
	 	done
	done
done
