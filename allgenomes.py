""" Class that keeps track of all genomes trained so far, and their scores.
    Among other things, ensures that genomes are unique.
"""

import random
import logging

from genome import Genome

class AllGenomes():
    """Store all genomes
    """

    def __init__(self, firstgenome):
        """Initialize
        """

        self.population = []
        self.population.append(firstgenome)
        
    def add_genome(self, genome):
        """Add the genome to our population.
        """

        for i in range(0,len(self.population)):
            if (genome.hash == self.population[i].hash):
                logging.info("add_genome() ERROR: hash clash - duplicate genome")
                return False

        self.population.append(genome)

        return True
        
    def set_accuracy(self, genome):
        """Add the genome to our population.
        """
        
        for i in range(0,len(self.population)):
            if (genome.hash == self.population[i].hash):
                self.population[i].accuracy = genome.accuracy
                return
    
        logging.info("set_accuracy() ERROR: Genome not found")

    def is_duplicate(self, genome):
        """Add the genome to our population.
        """

        for i in range(0,len(self.population)):
            if (genome.hash == self.population[i].hash):
                return True
    
        return False

    def print_all_genomes(self):
        """Print out a genome.
        """

        for genome in self.population:
            genome.print_genome_ma()