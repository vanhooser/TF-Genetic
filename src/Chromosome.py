import tensorflow as tf

class LayerChromosome(object):
	def __init__(self, activationFunctions = [], layerDimensions = [], previousInput=None, previousWidth=-1):
		self.activationFunctions, self.layerDimensions, self.previousInput, self.previousWidth = activationFunctions, layerDimensions, previousInput, previousWidth

	def construct(self, variables=None):
		if (not self.previousInput) or (not self.previousWidth):
			raise Exception("Propagation Error", "Need input from previous layer and dimension to construct this layer")
		weights = self.weightsForChromosome() if not variables else variables
		if not len(variables) == len(self.activationFunctions):
			raise Exception("Invalid Activation Exception", "Count of activation functions does not match count of variables")
		return weights, tf.concat(0, [self.activationFunctions[idx](variables[idx]) for idx in range(len(self.activationFunctions))])  # START HERE

	def weightsForChromosome(self):
		return [tf.Variable(tf.zeros(self.layerDimensions[idx])) for idx in range(len(self.layerDimensions))]
