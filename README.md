# TF-Genetic
###Evolutionary Neural Networks, backed by TensorFlow and pure Python

_Based on the work of Maul et al in [this](http://www.scs-europe.net/dlib/2014/ecms14papers/is_ECMS2014_0035.pdf) paper_

![Evolution](https://github.com/thepropterhoc/TF-Genetic/blob/master/src/images/evolution_1.gif)

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


- Import dependencies
```python
import tensorflow as tf
import numpy as np
import TFGenetic as gen

# Use TensorFlow's built-in Iris dataset
ds = tf.contrib.learn.datasets.base.load_iris()
ins = ds.data

# Auto-encode the data
outs = ds.data

# Declare valid activation functions for the network, and their corresponding colors for plotting
validActivationFunctions = [tf.nn.sigmoid, tf.nn.tanh, tf.nn.relu, tf.nn.softsign, tf.nn.elu]
activationFunctionColors = ['g', 'r', 'b', 'y', 'c']

# Declare the genetic pool and initialize properties
g = gen.GeneticPool(populationSize = 30, 
	tournamentSize = 4,
	memberDimensions = [4, 10, 10, 4], 
	mutationRate = 0.05,
	averagesCount = 2,
	validActivationFunctions = validActivationFunctions,
	activationFunctionColors = activationFunctionColors,
	ins = ins,
	outs = outs)

# Generate population and train
g.generatePopulation()

generationCount = 30
for generationNumber in range(generationCount):
	g.cycle()
	g.generation(generationNumber)
g.plotEvolution()
```

- For number of generations specified, cycle and generate new individuals
```python
numGenerations = 100
for _ in range(numGenerations):
	g.cycle()
	g.generation()
g.plotEvolution()
```
