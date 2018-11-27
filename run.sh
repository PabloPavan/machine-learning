

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
	python3 main_ionosphere.py net_io.txt | awk -vORS=, '{ print $2 }' | sed 's/,$/\n/'
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

echo "lambda,neural,alpha,media,variancia,desvio_padrao,tempo" > saida.txt

for lambda in 0 0.1; do 
	for alpha in 0.001 0.01; do 
		for layers in 1 4; do
	 		for neuron in 2 16 ; do 
	 	    	wrtitefile $lambda $alpha $layers $neuron
	 	    	run 
	 		done
	 	done
	done
done



#run comands to csv
#python3 main_ionosphere.py net_io.txt | awk -vORS=, '{ print $2 }' | sed 's/,$/\n/'
#0.01,0.01,0.387637362637,0.01313066809564061,0.11458912730115632,0.216017484664917