import random
import logging
import hashlib


children = []

        #where do we recombine? 0, 1, 2, 3, 4?
        #with four genes, there are three choices for the recombination
        #0 and 4 just (re)create more copies of the parents
        # ___ * ____ * _____ * ____
#recomb_loc = random.randint(1,3) 

#print("rl ", recomb_loc)

        #for _ in range(2): #make _two_ children - could also make more?

child1 = {} # {} = create empty dictionary
child2 = {}

all_possible_genes = {
    'optimizer':  ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam'],
    'nb_neurons': [64, 128, 256, 512, 768, 1024],
    'nb_layers':  [1, 2, 3, 4], # only relevant to mlp
    'activation': ['relu', 'elu', 'tanh', 'sigmoid']
}

#print(all_possible_genes['nb_layers'])
#print(len(all_possible_genes))

#0 -> no recombination, and N == length of dictionary -> no recombination
#so the range is always 1 to len(all_possible_genes) - 1
recomb_loc = random.randint(1,(len(all_possible_genes) - 1)) 
print("rl:", recomb_loc)

#keys = ['nb_neurons', 'nb_layers', 'activation', 'optimizer']
#we can't make assumptions about the length/nature of the dictionary
kn = list(all_possible_genes)
print("keys bf:", kn)
kn = sorted(kn)
print("keys as:", kn)

print("keys[]:", kn[0])

keys = ['nb_neurons', 'nb_layers', 'activation', 'optimizer']

print("keys[]:", keys[0])

print("keys[] b:", keys)
print("keys[] s:", keys.sort())

mom = {'nb_neurons': 128, 'nb_layers': 4, 'activation': 'tanh', 'optimizer': 'adadelta'}
dad = {'nb_neurons':  64, 'nb_layers': 2, 'activation': 'sigm', 'optimizer': 'bedelta'}

print(str(dad['nb_neurons'])+dad['activation'])
print(str(mom['nb_neurons'])+mom['activation'])

gen = str(dad['nb_neurons'])+dad['activation']
hash_object = hashlib.md5(gen)
print(hash_object.hexdigest())

gen = str(mom['nb_neurons'])+mom['activation']
hash_object = hashlib.md5(gen)
print(hash_object.hexdigest())


for x in range(0, len(all_possible_genes)):
    if x < recomb_loc:
        child1[keys[x]] = mom[keys[x]]#mom.genome[keys[x-1]]
        child2[keys[x]] = dad[keys[x]]#dad.genome[keys[x-1]]
    else:
        child1[keys[x]] = dad[keys[x]]#dad.genome[keys[x-1]]
        child2[keys[x]] = mom[keys[x]]#mom.genome[keys[x-1]]

print(mom)
print(dad)
print(child1)
print(child2)

#for key in all_possible_genes:
#    print(random.choice(all_possible_genes[key]))
#    child1[key] = random.choice(all_possible_genes[key])

#print("keys[0]: ", keys[0])

#print(child1[keys[0]])

# nb_layers  = genome['nb_layers']
# nb_neurons = genome['nb_neurons']
# activation = genome['activation']
# optimizer  = genome['optimizer']

#            all_possible_genes (dict): Parameters for the genome, includes:
#                gene_nb_neurons (list): [64, 128, 256]
#                gene_nb_layers (list):  [1, 2, 3, 4]
#                gene_activation (list): ['relu', 'elu']
#                gene_optimizer (list):  ['rmsprop', 'adam']

# is there much more elegant way to do this - yes of course. 

#        if recomb_loc == 1:
#            child1['nb_layers' ] = mom.genome['nb_layers' ]
#            child1['nb_neurons'] = dad.genome['nb_neurons']
#            child1['activation'] = dad.genome['activation']
#            child1['optimizer' ] = dad.genome['optimizer' ]