# -*- coding: utf-8 -*-
"""#TF-Genetic : genetic algorithm optimization of neural network structures

- This module is used to create and run a genetic algorithm on
top of TensorFlow-backed neural networks in order to find the
optimal structure and activation functions for any given problem.

"""

import tensorflow as tf
import TFGenetic as gen

TESTING = True

def run_tests():
    """#Tests

    _Run the genetic algorithm testing process_

    Loads the Iris dataset and tries the genetic algorithm
    """

    # Use TensorFlow's built-in Iris dataset
    iris_dataset = tf.contrib.learn.datasets.base.load_iris()
    input_data = iris_dataset.data

    # Auto-encode the data
    output_data = iris_dataset.data

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

    # Generate population and train
    genetic_pool.generatePopulation()

    generation_count = 5
    for generation_number in range(generation_count):
        genetic_pool.cycle()
        genetic_pool.generation(generation_number)
    genetic_pool.printAllSeenChromosomes()
    genetic_pool.plotEvolution()

if __name__ == '__main__':
    if TESTING:
        run_tests()
    else:
        print("Nothing to run")
