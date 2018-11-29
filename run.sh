#!/bin/bash

# lambda: 0.01
# rede: [33, 4, 2]
# alpha: 0.01
# media: 0.313624338624
# variancia: 0.019042371784104582
# desvio_padrao: 0.13799410054094552
# tempo: 0.21584486961364746

#create csv

#echo"lambda,rede,alpha,media,variancia,desvio_padrao,tempo" > output.csv

#create file of network

run ()
{
	HOST=`echo $HOSTNAME`
	OUTUPT=`python3 main_ionosphere.py net.txt | awk -vORS=, '{ print $2 }' | sed 's/,$/\n/'` 
	echo "$1,$2,"$OUTUPT >> $HOST"_"$DATE"_"ionosphere.csv
}

wrtitefile () 
{ 
	echo $1 > net.txt
	echo $2 >> net.txt
	echo "33" >> net.txt
	for a in `seq 1 $3`; do
		echo $4 >> net.txt
	done
	echo "2" >> net.txt
} 

HOST=`hostname`
DATE=`date "+%d%m%Y-%H%M%S"`

echo "layers,neuron,lambda,alpha,f1_mean,variance,standard_deviation,time_execution" > $HOST"_"$DATE"_"ionosphere.csv

for lambda in 0 0.1; do
	for alpha in 0.001 0.01; do
		for layers in 1; do
	 		for neuron in 2 16 ; do 
	 	    	wrtitefile $lambda $alpha $layers $neuron
	 	    	run $layers $neuron
	 		done
	 	done
	done
done

