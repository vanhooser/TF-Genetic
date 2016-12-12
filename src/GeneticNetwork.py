import tensorflow as tf
import random

class GeneticNetwork(object):
	def __init__(self, layerDimensions=[], netDimensions=[], validActivationFunctions=[]):

		self.layerDimensions = layerDimensions
		
		self.x = tf.placeholder(tf.float32, [None, netDimensions[0]])
		previousActivation = self.x

		for idx in range(len(layerDimensions)):
			currentLayer = layerDimensions[idx]
			thisActivation = None
			for functionIndex in range(len(currentLayer)):
				inDim, outDim = currentLayer[functionIndex]
				thisW = tf.Variable(tf.random_normal([inDim, outDim]))
				thisB = tf.Variable(tf.random_normal([outDim]))
				thisFunction = validActivationFunctions[functionIndex]
				newTensor = thisFunction(tf.matmul(previousActivation, thisW) + thisB)
				thisActivation = newTensor if thisActivation is None else tf.concat(1, [thisActivation, newTensor])
					
			previousActivation = thisActivation

		self.predictedOutput = previousActivation
		self.y_ = tf.placeholder(tf.float32, [None, netDimensions[-1]])
		cross_entropy = tf.reduce_mean(tf.square(self.predictedOutput - self.y_))
		self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
		
		init = tf.initialize_all_variables()
		self.sess = tf.Session(config=tf.ConfigProto(
            inter_op_parallelism_threads=4
        ))
		self.sess.run(init)
		
	def trainStep(self, ins=[], outs=[], chooseCount=-1):
		if chooseCount == -1:
			batch_xs, batch_ys = ins, outs
		else:
			chosenIndices = [random.randint(0, len(ins)-1) for _ in range(chooseCount)]
			batch_xs, batch_ys = list(zip(*[(ins[x], [outs[x]]) for x in chosenIndices]))

		cross_entropy = tf.reduce_mean(tf.square(self.predictedOutput - self.y_))

		_, loss = self.sess.run([self.train_step, cross_entropy], feed_dict={self.x: batch_xs, self.y_: batch_ys})
		return loss

	def close(self):
		self.sess.close()