"""Entry point to evolving the neural network. Start here."""
from __future__ import print_function

from evolver import Evolver

from tqdm import tqdm

import logging

import sys

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO#,
    #filename='log.txt'
)

def train_genomes(genomes, dataset):
    """Train each genome.

    Args:
        networks (list): Current population of genomes
        dataset (str): Dataset to use for training/evaluating

    """
    logging.info("***train_networks(networks, dataset)***")

    pbar = tqdm(total=len(genomes))

    for genome in genomes:
        genome.train(dataset)
        pbar.update(1)
    
    pbar.close()

def get_average_accuracy(genomes):
    """Get the average accuracy for a group of networks/genomes.

    Args:
        networks (list): List of networks/genomes

    Returns:
        float: The average accuracy of a population of networks/genomes.

    """
    total_accuracy = 0

    for genome in genomes:
        total_accuracy += genome.accuracy

    return total_accuracy / len(genomes)

def generate(generations, population, all_possible_genes, dataset):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evolve the population
        population (int): Number of networks in each generation
        all_possible_genes (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating

    """
    logging.info("***generate(generations, population, all_possible_genes, dataset)***")
    
    evolver = Evolver(all_possible_genes)
    
    genomes = evolver.create_population(population)

    # Evolve the generation.
    for i in range( generations ):

        logging.info("***Now in generation %d of %d***" % (i + 1, generations))

        print_genomes(genomes)
        
        # Train and get accuracy for networks/genomes.
        train_genomes(genomes, dataset)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(genomes)

        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-'*80) #-----------

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Evolve!
            genomes = evolver.evolve(genomes)

    # Sort our final population according to performance.
    genomes = sorted(genomes, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks/genomes.
    print_genomes(genomes[:5])

    #save_path = saver.save(sess, '/output/model.ckpt')
    #print("Model saved in file: %s" % save_path)

def print_genomes(genomes):
    """Print a list of genomes.

    Args:
        genomes (list): The population of networks/genomes

    """
    logging.info('-'*80)

    for genome in genomes:
        genome.print_genome()

def main():
    """Evolve a genome."""
    population = 30 # Number of networks/genomes in each generation.
    #we only need to train the new ones....
    
    ds = ''

    if (ds == 1):
        dataset = 'mnist_mlp'
    elif (ds == 2):
        dataset = 'mnist_cnn'
    elif (ds == 3):
        dataset = 'cifar10_mlp'
    elif (ds == 4):
        dataset = 'cifar10_cnn'
    elif (ds == 5):
        dataset = 'mnist_mlp'
    else:
        dataset = 'custom_cnn'

    print("***Dataset:", dataset)

    if dataset == 'mnist_cnn':
        generations = 8 # Number of times to evolve the population.
        all_possible_genes = {
            'nb_neurons': [16, 32, 64, 128],
            'nb_layers':  [1, 2, 3, 4 ,5],
            'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'],
            'optimizer':  ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam']
        }
    elif dataset == 'mnist_mlp':
        generations = 8 # Number of times to evolve the population.
        all_possible_genes = {
            'nb_neurons': [64, 128], #, 256, 512, 768, 1024],
            'nb_layers':  [1, 2, 3, 4, 5],
            'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'],
            'optimizer':  ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam']
        }
    elif dataset == 'cifar10_mlp':
        generations = 8 # Number of times to evolve the population.
        all_possible_genes = {
            'nb_neurons': [64, 128, 256, 512, 768, 1024],
            'nb_layers':  [1, 2, 3, 4, 5],
            'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'],
            'optimizer':  ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam']
        }
    elif dataset == 'cifar10_cnn':
        generations = 8 # Number of times to evolve the population.
        all_possible_genes = {
            'nb_neurons': [16, 32, 64, 128],
            'nb_layers':  [1, 2, 3, 4, 5],
            'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'],
            'optimizer':  ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam']
        }
    elif dataset == 'custom_cnn':
        generations = 8 # Number of times to evolve the population.
        all_possible_genes = {
            'nb_neurons': [16, 32, 64, 128],
            'nb_layers':  [1, 2, 3, 4, 5],
            'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'],
            'optimizer':  ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam']
        }
    else:
        generations = 8 # Number of times to evolve the population.
        all_possible_genes = {
            'nb_neurons': [64, 128, 256, 512, 768, 1024],
            'nb_layers':  [1, 2, 3, 4, 5],
            'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'],
            'optimizer':  ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam']
        }
            
    print("***Evolving for %d generations with population size = %d***" % (generations, population))

    generate(generations, population, all_possible_genes, dataset)

if __name__ == '__main__':
    main()
