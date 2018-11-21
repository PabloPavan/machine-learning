import numpy as np
import re
import math
from utils import *
from nn import *
import argcomplete,argparse
import logging
import logging.handlers
from argcomplete.completers import EnvironCompleter

def read_files(network, weights):
    
    return np.loadtxt(network, dtype='float', delimiter=',', skiprows=0)[0],  np.loadtxt(network, dtype='i', delimiter=',', skiprows=1), open(weights, "r")

def main():

    parser = argparse.ArgumentParser(description='Argument')
    parser.add_argument('--network', "--n",required=True,  metavar='FILE', type=str).completer = EnvironCompleter
    parser.add_argument('--weights', "--w",required=True,  metavar='FILE', type=str).completer = EnvironCompleter
    parser.add_argument('--dataset', "--d",required=True,  metavar='FILE', type=str).completer = EnvironCompleter
    parser.add_argument('--log', "--l", required=False, default=True).completer = EnvironCompleter

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    if args.log:
        configureLog(logger=logger)
     
    # init 
    reg_value, network, initial_weights_file = read_files(args.network, args.weights)
    logger.info('network format: {}'.format(network)) 

    nn = neural_network(reg_value, network, initial_weights_file, args.dataset, logger)


    nn.create_vectors()
    # for kfold 
    nn.training()

    
    print("\nJ", nn.j_total)


if __name__ == "__main__":
    logger = logging.getLogger('neural_network')
    main()