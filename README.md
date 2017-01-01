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
```

- Define valid activation functions the genetic algorithm will evolve
```python
validActivationFunctions = [tf.nn.sigmoid, tf.nn.tanh, tf.nn.relu, tf.nn.softsign]
```

- Initialize a genetic algorithm population and describe the initial structure of the population dimensions.  Here with the Iris dataset, the network is a 4 -> x -> 1 network type
```python	
g = gen.GeneticPool(
 			populationSize = 10, 
			tournamentSize = 4,
			memberDimensions = [4, 10, 5, 1], 
			validActivationFunctions = validActivationFunctions
			)
g.generatePopulation()
```

- For number of generations specified, cycle and generate new individuals
```python
numGenerations = 100
for _ in range(numGenerations):
	g.cycle()
	g.generation()
g.plotEvolution()
```
