import tensorflow as tf
import Chromosome as chrom
import TFGenetic as gen
import random


def experiments():

	# Data sets
	IRIS_TRAINING = "iris_training.csv"
	IRIS_TEST = "iris_test.csv"

	

	print(ins)

	netDimensions = [4, 10, 5, 1]

	x = tf.placeholder(tf.float32, [None, netDimensions[0]])

	previousActivation = x

	for idx in range(1, len(netDimensions)):
			
		# Weights ingest dimensions of previous layer and output the current dimension 
		thisW = tf.Variable(tf.zeros([netDimensions[idx-1], netDimensions[idx]]))
		thisB = tf.Variable(tf.zeros([netDimensions[idx]]))
		thisActivation = tf.nn.sigmoid(tf.matmul(previousActivation, thisW) + thisB)
		previousActivation = thisActivation

	predictedOutput = previousActivation
	y_ = tf.placeholder(tf.float32, [None, netDimensions[-1]])
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ - predictedOutput, reduction_indices=[1]))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	

	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)

	chooseCount = 50

	for step in range(1000):

		chosenIndices = [random.randint(0, len(ins)-1) for _ in range(chooseCount)]
		batch_xs, batch_ys = list(zip(*[(ins[x], [outs[x]]) for x in chosenIndices]))

		cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ - predictedOutput, reduction_indices=[1]))

		_, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})

		if step % 100 == 0:
			print(loss)
	"""
	activationFunctionDimensions = [[20, 1], [20, 1], [5,1]]
	c_activated = chrom.LayerChromosome(activationFunctions = validActivationFunctions, layerDimensions=activationFunctionDimensions).construct()
	print(c_activated)
	"""


if __name__ == '__main__':

	testing = True

	if testing:

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
	else:
		experiments()


