# DeepEvolve 

These days, it's relatively easy to *train* neural networks, but it's still difficult to figure out which network architectures and other hyperparameters to use - e.g. how many neurons, how many layers, and which activation functions? In the long term, of course, neural networks will learn how to architect themselves, without human intervention. Until then, the speed of developing application-optimized neural networks will remain limited by the time and expertise required to chose and refine hyperparameters. DeepEvolve is designed to help solve this problem, by rapidly returning good hyperparameters for particular datasets and classification problems. The code supports hyperparameter discovery for MLPs (ie. fully connected networks) and convolutional neural networks.

If you had infinite time and infinite computing resources, you could brute-force the problem, and just compare and contrast all parameter combinations. However, in most real-world applications of neural networks, you will probably have to balance competing demands (time, cost, desire to continuously optimize AI performance in dynamic environments) and you may - for whatever reason - have a strong interest to be able to rapidly generate good networks for diverse datasets. In that case, genetic algorithms will be useful.  

## Genetic Algorithms

Genetic algorithms can be used to solve complex nonlinear optimization problems. DeepEvolve is a simple Keras framework for rapidly discovering good hyperparameters using cycles of mutation, recombination, training, and selection. The role of **point mutations** in genomes is readily apparent - create diversity - but the functions of other genome operations, such as **recombination**, are not as widely appreciated. Briefly, recombination addresses [**clonal interference**](https://en.wikipedia.org/wiki/Clonal_interference), which is a major kinetic bottleneck in discovering optimal genomes in evolving populations. 

Imagine that two (or more) *different* beneficial mutations in *different* genes arise independently in different individuals. These individuals will have higher fitness, but the algorithm (aka evolution) can not easily converge on the optimal genome. Evolution solves clonal interference through recombination, which allows two genomes to swap entire regions, increasing the likelihood of generating a single genome with *both* beneficial genes. If you are curious about clonal interference, a good place to start is [*The fate of competing beneficial mutations in an asexual population*](https://link.springer.com/article/10.1023%2FA%3A1017067816551). 

## History of the codebase

DeepEvolve is based on [Matt Harvey's Keras code](https://github.com/harvitronix/neural-network-genetic-algorithm), which in turns draws from [Will Larson, Genetic Algorithms: Cool Name & Damn Simple](https://lethain.com/genetic-algorithms-cool-name-damn-simple/).

## Important aspects of the code

Each AI network architecture is represented as a string of genes. These architectures/genomes recombine with some frequency, at one randomly selected position along the genome. Note that a genome with *N* genes can recombine at *N* - 1 nontrivial positions (1, 2, 3, N-1). Specifically, ```recomb_loc = 0 || len(self.all_possible_genes)``` does not lead to recombination, but just returns the original parental genomes, and therefore ```recomb_loc = random.randint(1, len(self.all_possible_genes) - 1)```. 

```python
recomb_loc = random.randint(1,len(self.all_possible_genes) - 1) 

keys = list(self.all_possible_genes)

*** CORE RECOMBINATION CODE ****
for x in range(0, pcl):
    if x < recomb_loc:
        child1[keys[x]] = mom.geneparam[keys[x]]
        child2[keys[x]] = dad.geneparam[keys[x]]
    else:
        child1[keys[x]] = dad.geneparam[keys[x]]
        child2[keys[x]] = mom.geneparam[keys[x]]
```

To increase the rate of discovering optimal hyperparameters, we also keep track of all genomes in all previous generations; each genome is identified via its MD5 hash and we block recreation of duplicate, previously generated and trained genomes.  

```python
if self.random_select > random.random():
    gtc = copy.deepcopy(genome)
                
    while self.master.is_duplicate(gtc):
        gtc.mutate_one_gene()
```

Finally, we also facilitate genome uniqueness during the mutation operation, by limiting random choices to ones that differ from the gene's current value. 

```python
def mutate_one_gene(self):

    # Which gene shall we mutate? Choose one of N possible keys/genes.
    gene_to_mutate = random.choice( list(self.all_possible_genes.keys()) )

    # Make sure that we actually create a new variant
    current_value    = self.geneparam[gene_to_mutate]
    possible_choices = copy.deepcopy(self.all_possible_genes[gene_to_mutate])    
    possible_choices.remove(current_value)
            
    self.geneparam[gene_to_mutate] = random.choice( possible_choices )

    self.update_hash()
```

If you try to keep all the models in memory, you will quickly run out, so we use ```K.clear_session()``` after every training run. 

## To run

To run the brute force algorithm which goes through all possible choices one by one:

```python3 brute.py```

To run the genetic algorithm:

```python3 main.py```

In general, you will want to run the code in the cloud - we use [floydhub.com](http:floydhub.com):

```$ floyd run --gpu --env keras "python main.py"```

For a convolutional neural network being trained on `cifar10`, expect run times of about 3.2 hours on a Tesla K80 (30 genomes and 8 generations). Compared to the brute force solution, you should expect to get high performing hyperparameter combinations within about 3.5 generations, about 8x to 12x faster than brute force. Of course, with a genetic algorithm, you are not *guaranteed* to find the best solution, you are merely likely to find something suitable relatively quickly. You can choose the various options in ```main.py```.

## Examples

Here are the time dynamics of discovering the optimal optimizer (Adam). The x axis is time, and the y axis is gene variant frequency in the population. Adaptive Moment Estimation ([Adam](https://arxiv.org/pdf/1412.6980.pdf)) computes adaptive learning rates based on exponential moving averages of the 1st moment (the mean) and the 2nd raw moment (the uncentered variance) of the gradient. By the 5th generation, genomes with adaptive moment estimation are the dominant subspecies in the population. 

![alt text](https://github.com/jliphard/DeepEvolve/blob/726aaf3dfdc8d6d2c6bc64d3a55e3ab3023b29c7/Images/Optimizer.png "Optimizer kinetics")

In this figure, you can see a small selection of genomes being trained for 4 epochs on cifar10. The x axis is time, and the y axis is accuracy. Some genomes fail to learn at all (flat line at 10% accuracy), while about 1/2 of the genomes achieve 35% accuracy within one epoch. Early performance does not perfectly predict learning rates in subsequent epochs, but it comes close. 

![alt text](https://github.com/jliphard/DeepEvolve/blob/4f8cf547797b2263659f053e0824bf34b39e337a/Images/Evolve.png "Evolution kinetics")

This is a graphical overview of how the AI hyperparameter sets (aka genomes) evolve, showing the initial (random) population of genomes (left), and the various cycles of recombination and selection. 

![alt text](https://github.com/jliphard/DeepEvolve/blob/55473015692e2af75be35fa1baf6536e300032bc/Images/Network.png "Evolution of AI hyperparameter sets")
