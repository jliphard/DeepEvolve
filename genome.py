"""The genome to be evolved."""

import random
import logging
import hashlib
import copy

from train import train_and_score

class Genome():
    """
    Represents one genome and all relevant utility functions (add, mutate, etc.).
    """

    def __init__( self, all_possible_genes = None, geneparam = {}, u_ID = 0, mom_ID = 0, dad_ID = 0, gen = 0 ):
        """Initialize a genome.

        Args:
            all_possible_genes (dict): Parameters for the genome, includes:
                gene_nb_neurons_i (list): [64, 128, 256]      for (i=1,...,6)
                gene_nb_layers (list):  [1, 2, 3, 4]
                gene_activation (list): ['relu', 'elu']
                gene_optimizer (list):  ['rmsprop', 'adam']
        """
        self.accuracy         = 0.0
        self.all_possible_genes = all_possible_genes
        self.geneparam        = geneparam #(dict): represents actual genome parameters
        self.u_ID             = u_ID
        self.parents          = [mom_ID, dad_ID]
        self.generation       = gen
        
        #hash only makes sense when we have specified the genes
        if not geneparam:
            self.hash = 0
        else:
            self.update_hash()
        
    def update_hash(self):
        """
        Refesh each genome's unique hash - needs to run after any genome changes.
        """
        genh = str(self.nb_neurons()) + self.geneparam['activation'] \
                + str(self.geneparam['nb_layers']) + self.geneparam['optimizer']

        self.hash = hashlib.md5(genh.encode("UTF-8")).hexdigest()

        self.accuracy = 0.0
            
    def set_genes_random(self):
        """Create a random genome."""
        #print("set_genes_random")
        self.parents = [0,0] #very sad - no parents :(

        for key in self.all_possible_genes:
            self.geneparam[key] = random.choice(self.all_possible_genes[key])
                
        self.update_hash()
        
    def mutate_one_gene(self):
        """Randomly mutate one gene in the genome.

        Args:
            network (dict): The genome parameters to mutate

        Returns:
            (Genome): A randomly mutated genome object

        """
        # Which gene shall we mutate? Choose one of N possible keys/genes.
        gene_to_mutate = random.choice( list(self.all_possible_genes.keys()) )

        # And then let's mutate one of the genes.
        # Make sure that this actually creates mutation
        current_value    = self.geneparam[gene_to_mutate]
        possible_choices = copy.deepcopy(self.all_possible_genes[gene_to_mutate])
        
        possible_choices.remove(current_value)
        
        self.geneparam[gene_to_mutate] = random.choice( possible_choices )

        self.update_hash()
    
    def set_generation(self, generation):
        """needed when a genome is passed on from one generation to the next.
        the id stays the same, but the generation is increased"""   

        self.generation = generation
        #logging.info("Setting Generation to %d" % self.generation)

    def set_genes_to(self, geneparam, mom_ID, dad_ID):
        """Set genome properties.
        this is used when breeding kids

        Args:
            genome (dict): The genome parameters
        IMPROVE
        """
        self.parents  = [mom_ID, dad_ID]
        
        self.geneparam = geneparam

        self.update_hash()

    def train(self, trainingset):
        """Train the genome and record the accuracy.

        Args:
            dataset (str): Name of dataset to use.

        """
        if self.accuracy == 0.0: #don't bother retraining ones we already trained 
            self.accuracy = train_and_score(self, trainingset)

    def print_genome(self):
        """Print out a genome."""
        self.print_geneparam()
        logging.info("Acc: %.2f%%" % (self.accuracy * 100))
        logging.info("UniID: %d" % self.u_ID)
        logging.info("Mom and Dad: %d %d" % (self.parents[0], self.parents[1]))
        logging.info("Gen: %d" % self.generation)
        logging.info("Hash: %s" % self.hash)

    def print_genome_ma(self):
        """Print out a genome."""
        self.print_geneparam()
        logging.info("Acc: %.2f%% UniID: %d Mom and Dad: %d %d Gen: %d" % (self.accuracy * 100, self.u_ID, self.parents[0], self.parents[1], self.generation))
        logging.info("Hash: %s" % self.hash)

    # print nb_neurons as single list
    def print_geneparam(self):
        g = self.geneparam.copy()
        nb_neurons = self.nb_neurons()
        for i in range(1,7):
          g.pop('nb_neurons_' + str(i))
        # replace individual layer numbers with single list
        g['nb_neurons'] = nb_neurons
        logging.info(g)
    
    # convert nb_neurons_i at each layer to a single list
    def nb_neurons(self):
      nb_neurons = [None] * 6
      for i in range(0,6):
        nb_neurons[i] = self.geneparam['nb_neurons_' + str(i+1)]

      return nb_neurons
