import matplotlib
matplotlib.use('tkAgg')

import tensorflow as tf
import Chromosome as chrom
import GeneticNetwork as gn
import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import multiprocessing

class GeneticPool(object):
	def __init__(
		self, 
		populationSize = 10, 
		tournamentSize = 2, 
		mutationRate = 0.1, 
		memberDimensions = [5,5,1], 
		validActivationFunctions = [tf.nn.sigmoid],
		activationFunctionColors = [],
		numEpochs = 500,
		averagesCount = 3,
		ins = None,
		outs = None):

		self.populationSize = populationSize
		self.tournamentSize = tournamentSize
		self.mutationRate = mutationRate
		self.memberDimensions = memberDimensions
		self.validActivationFunctions = validActivationFunctions
		self.numEpochs = numEpochs
		self.activationFunctionColors = activationFunctionColors
		self.averagesCount = averagesCount

		self.evolutionPlot = []

		if ins is None and outs is None:
			ds = tf.contrib.learn.datasets.base.load_iris()
			self.ins = ds.data
			self.outs = ds.target
		else:
			self.ins, self.outs = ins, outs

	def generatePopulation(self):
		self.chromosomes = []
		for _ in range(self.populationSize):
			c = self.generateChromosome()
			self.chromosomes += [c]

	def dimensionsForStructure(self, networkDimensions=[]):
		returnedDimensions = [networkDimensions[0][0][0]]
		for layer in networkDimensions:
			currentOutputDimension = 0
			for (inputDim, outputDim) in layer:
				currentOutputDimension += outputDim
			returnedDimensions += [currentOutputDimension]
		return returnedDimensions

	def draw_neural_net(self, ax, left, right, bottom, top, layer_sizes):
			'''
			Draw a neural network cartoon using matplotilb.
			
			:usage:
					>>> fig = plt.figure(figsize=(12, 12))
					>>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
			
			:parameters:
					- ax : matplotlib.axes.AxesSubplot
							The axes on which to plot the cartoon (get e.g. by plt.gca())
					- left : float
							The center of the leftmost node(s) will be placed here
					- right : float
							The center of the rightmost node(s) will be placed here
					- bottom : float
							The center of the bottommost node(s) will be placed here
					- top : float
							The center of the topmost node(s) will be placed here
					- layer_sizes : list of int
							List of layer sizes, including input and output dimensionality
			'''

			n_layers = len(layer_sizes) + 1
			maxLayerSize = 0

			for layer_group in layer_sizes:
				layer_size = sum(map(lambda x: x[1], layer_group))
				maxLayerSize = max(maxLayerSize, layer_size)
			v_spacing = (top - bottom)/float(maxLayerSize)
			h_spacing = (right - left)/float(len(layer_sizes) - 1)

			# Nodes
			for layerIndex, layer_group in enumerate(layer_sizes):
				layer_size = sum(map(lambda x: x[1], layer_group))
				layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
				currentIndex = 0
				for functionIndex, (inputSize, outputSize) in enumerate(layer_group):

					if outputSize > 0:
						
						for nodeIndex in range(outputSize):
							circle = plt.Circle((layerIndex*h_spacing + left, layer_top - currentIndex*v_spacing), v_spacing/4.,
																	color=self.activationFunctionColors[functionIndex], ec='k', zorder=4)
							ax.add_artist(circle)
							currentIndex += 1
					else:
						# Null nodes, ignore and keep going 
						continue
							
			"""
			# Edges
			for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
					layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
					layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
					for m in xrange(layer_size_a):
							for o in xrange(layer_size_b):
									line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
																		[layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
									ax.add_artist(line)
			"""


	def chromosomeRun(self, c):
		dims = self.dimensionsForStructure(c)
		totalLoss = 0.0
		for _ in range(self.averagesCount):
			member = gn.GeneticNetwork(c, dims, self.validActivationFunctions)
			finalLoss = 0.0
			for epoch in range(self.numEpochs):
				thisLoss = member.trainStep(ins=self.ins, outs=self.outs)
				if epoch == self.numEpochs - 1:
					finalLoss = thisLoss
				else:
					continue
			member.close()
			totalLoss += finalLoss
		return totalLoss / float(self.averagesCount)

	def cycle(self):
		with multiprocessing.Pool(4) as p:
			self.fitnesses = p.map(self.chromosomeRun, self.chromosomes)

	def generation(self, generationNumber):
		zippedFitnesses = sorted(zip(self.fitnesses, self.chromosomes), key=lambda x: x[0])


		currentTotalFitness = 0.0
		for f, c in zippedFitnesses:
			currentTotalFitness += f

		self.evolutionPlot += [currentTotalFitness]
		print("Current total Error : ", currentTotalFitness)

		bestFitness, bestChromosome = zippedFitnesses[0]
		newChromosomes = [list(bestChromosome)]

		print("Current best fitness : ", bestFitness)

		fig = plt.figure(figsize=(12, 12))
		ax = fig.gca()
		ax.axis('off')
		self.draw_neural_net(ax, .1, .9, .1, .9, bestChromosome)
		fig.savefig('./images/best_{0}.png'.format(generationNumber))

		for tournamentNumber in range(self.populationSize - 1):
			m1, m2 = self.tournamentSelect(zippedFitnesses)
			child = self.crossover(m1, m2)
			newChromosomes += [child]

		for idx in list(range(len(newChromosomes)))[1:]:
			if random.random() < self.mutationRate:
				newChromosomes[idx] = self.generateChromosome()
				print("MUTATION at index ", idx, " to chromosome ", newChromosomes[idx])
		self.chromosomes = newChromosomes
		return

	def tournamentSelect(self, zippedFitnesses=[]):
		selected = [zippedFitnesses[x] for x in np.random.choice(len(zippedFitnesses), self.tournamentSize, replace=False).tolist()]
		tournamentSelected = sorted(selected, key = lambda x: x[0])
		selectedIndex1 = self.selectOne(tournamentSelected)
		m1 = tournamentSelected[selectedIndex1]
		del tournamentSelected[selectedIndex1]
		selectedIndex2 = self.selectOne(tournamentSelected)
		m2 = tournamentSelected[selectedIndex2]
		return m1, m2


	def selectOne(self, tournamentSelected):
		totalFitness = 0.0
		for x in tournamentSelected:
			totalFitness += float(x[0])
		selectedFitness = random.random() * totalFitness
		for idx in range(len(tournamentSelected)):
			selectedFitness -= tournamentSelected[idx][0]
			if selectedFitness <= 0:
				return idx
		return len(tournamentSelected) - 1

	def crossover(self, m1, m2):
		(f1, c1), (f2, c2) = list(m1), list(m2)
		totalFitness = f1 + f2
		returnedChromosome = []
		previousOutputDimension = c1[0][0][0]
		for layerIndex in range(len(c1) - 1):
			c1Layer, c2Layer = c1[layerIndex], c2[layerIndex]
			layerChromosome = []
			currentOutputDimension = 0
			for functionIndex in range(len(c1Layer)):
				(c1i, c1o), (c2i, c2o) = tuple(c1Layer[functionIndex]), tuple(c2Layer[functionIndex])
				selectedRandom = random.random() * totalFitness
				if selectedRandom - f1 <= 0:
					# Select c1 point function
					layerChromosome += [[previousOutputDimension, c1o]]
					currentOutputDimension += c1o
				else:
					layerChromosome += [[previousOutputDimension, c2o]]
					currentOutputDimension += c2o
					# Select c2 point function
			returnedChromosome += [layerChromosome]
			previousOutputDimension = currentOutputDimension
		finalLayer = None
		if f1 > f2:
			# Select final output from c1
			finalLayer = copy.deepcopy(c1[-1])

		else:
			# Select final output from c2
			finalLayer = copy.deepcopy(c2[-1])
		for idx in range(len(finalLayer)):
			finalLayer[idx][0] = previousOutputDimension
		returnedChromosome += [finalLayer]
		return returnedChromosome

	def generateChromosome(self):
		allLayerDimensions = []
		for layerIdx in range(1, len(self.memberDimensions)):
			layerDimension = self.memberDimensions[layerIdx]
			inputDimension = self.memberDimensions[layerIdx-1]
			currentLayerConsumedSize = 0
			activationFunctionDimensions = []
			for validActivationFunction in self.validActivationFunctions[:-1]:
				functionSize = random.randint(0, layerDimension - currentLayerConsumedSize)
				activationFunctionDimensions += [[inputDimension, functionSize]]
				currentLayerConsumedSize += functionSize
			functionSize = layerDimension - currentLayerConsumedSize
			activationFunctionDimensions += [[inputDimension, functionSize]]
			allLayerDimensions += [activationFunctionDimensions]			
		return allLayerDimensions

	def plotEvolution(self):
		plt.plot(self.evolutionPlot)
		plt.xlabel("Generation")
		plt.ylabel("Total Error")
		plt.title("Error by generation")
		plt.show()
