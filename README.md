# TF-Genetic
###Evolutionary Neural Networks, backed by TensorFlow and pure Python

_Based on the work of Maul et al in [this](http://www.scs-europe.net/dlib/2014/ecms14papers/is_ECMS2014_0035.pdf) paper_

![Evolution](https://github.com/thepropterhoc/TF-Genetic/blob/master/src/images/best_0.png)

## Description

This package speeds up the evolution of activation function structure in neural networks.  Nets can sometimes become even more accurate to their problem domain when activation functions within each layer are _mixed_ together and not uniformly applied to all neurons.  

This package allows for easy simulation of an arbitrary number of layers/neurons/activation functions to find an optimal arrangement. 

## Dependencies

- [TensorFlow](https://www.tensorflow.org/versions/r0.11/get_started/index.html) (tested 0.11.0rc0)
- [Numpy](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html) (tested 1.11.2)


## Structure

```
.
├── LICENSE
├── README.md
└── src
    ├── Chromosome.py
    ├── GeneticNetwork.py
    ├── TFGenetic.py
    └── tests.py
```

## Thirty Seconds to TF-Genetic

- Use TensorFlow's included Iris dataset
```python
# Use TensorFlow's built-in Iris dataset
iris_dataset = tf.contrib.learn.datasets.base.load_iris()
input_data = iris_dataset.data

# Auto-encode the data
output_data = iris_dataset.data
```

- Define valid activation functions for the evolution and colors for each to be plotted as
```python

# Declare valid activation functions for the network,
# and their corresponding colors for plotting
valid_activation_function_list = [
    tf.nn.sigmoid,
    tf.nn.tanh,
    tf.nn.relu,
    tf.nn.softsign,
    tf.nn.elu]

activation_function_colors = [
    'g',
    'r',
    'b',
    'y',
    'c']
    
```
    
- Generate a genetic algorithm pool and specify tournament size, population size, etc. 
```python

genetic_pool_settings = {
    'populationSize' : 30,
    'tournamentSize' : 4,
    'memberDimensions' : [4, 3, 2, 3, 4],
    'mutationRate' : 0.05,
    'averagesCount' : 1,
    'validActivationFunctions' : valid_activation_function_list,
    'activationFunctionColors' : activation_function_colors,
    'ins' : input_data,
    'outs' : output_data
}

# Declare the genetic pool and initialize properties
genetic_pool = gen.GeneticPool(**genetic_pool_settings)

# Generate population
genetic_pool.generatePopulation()
```

- For number of generations specified, cycle and generate new individuals
```python

generation_count = 5
for generation_number in range(generation_count):
    genetic_pool.cycle()
    genetic_pool.generation(generation_number)
genetic_pool.printAllSeenChromosomes()
genetic_pool.plotEvolution()
```

- Profit
