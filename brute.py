"""Iterate over every combination of hyperparameters."""
from __future__ import print_function
import logging
from genome import Genome
from tqdm import tqdm

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO#,
    #filename='brute-log.txt'
)

def train_genomes(genomes, dataset):
    """Train each network.

    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    pbar = tqdm(total=len(genomes))
    
    for genome in genomes:
        genome.train(dataset)
        genome.print_genome()
        pbar.update(1)
    pbar.close()

    # Sort our final population.
    genomes = sorted(genomes, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks.
    print_genomes(genomes[:5])

def print_genomes(genomes):
    """Print a list of networks.

    Args:
        networks (list): The population of networks

    """
    logging.info('-'*80)
    for genome in genomes:
        genome.print_genome()

def generate_genome_list(all_possible_genes):
    """Generate a list of all possible networks.

    Args:
        all_possible_genes (dict): The parameter choices

    Returns:
        networks (list): A list of network objects

    """
    genomes = []

    # This is silly.
    for nbn in all_possible_genes['nb_neurons']:
        for nbl in all_possible_genes['nb_layers']:
            for a in all_possible_genes['activation']:
                for o in all_possible_genes['optimizer']:

                    # Set the parameters.
                    genome = {
                        'nb_neurons': nbn,
                        'nb_layers': nbl,
                        'activation': a,
                        'optimizer': o,
                    }

                    # Instantiate a network object with set parameters.
                    genome_obj = Genome()
                    genome_obj.set_genes_to(genome, 0, 0)
                    genomes.append(genome_obj)

    return genomes

def main():
    """Brute force test every network."""
    dataset = 'cifar10_cnn'

    all_possible_genes = {
        'nb_neurons': [16, 32, 64, 128],
        'nb_layers':  [1, 2, 3, 4, 5],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'],
        'optimizer':  ['rmsprop', 'adam', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam'],
    }

    logging.info("***Brute forcing networks***")

    genomes = generate_genome_list(all_possible_genes)

    train_genomes(genomes, dataset)

if __name__ == '__main__':
    main()
