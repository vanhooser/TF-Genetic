import tensorflow as tf
import Chromosome as chrom
import GeneticNetwork as gn
import numpy as np
import random
import matplotlib
import copy
matplotlib.use('tkAgg')
from matplotlib import pyplot as plt

class GeneticPool(object):
	def __init__(self, 
		populationSize = 10, 
		tournamentSize = 2, 
		mutationRate = 0.1, 
		memberDimensions = [5,5,1], 
		validActivationFunctions = [tf.nn.sigmoid],
		numEpochs = 30):
		self.populationSize, self.tournamentSize, self.mutationRate, self.memberDimensions, self.validActivationFunctions, self.numEpochs = populationSize, tournamentSize, mutationRate, memberDimensions, validActivationFunctions, numEpochs
		self.evolutionPlot = []

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

	def cycle(self):
		self.fitnesses = []
		# Load datasets.
		ds = tf.contrib.learn.datasets.base.load_iris()
		ins = ds.data
		outs = ds.target

		for c in self.chromosomes:
			print(c)
			dims = self.dimensionsForStructure(c)
			member = gn.GeneticNetwork(c, dims, self.validActivationFunctions)
			finalLoss = 0.0
			for epoch in range(self.numEpochs):
				thisLoss = member.trainStep(ins=ins, outs=outs, chooseCount=50)
				if epoch == self.numEpochs - 1:
					finalLoss = thisLoss
				else:
					continue
					"""
					if epoch % int(float(self.numEpochs) / 10.0) == 0:
						print("Epoch ", epoch, " : ", thisLoss)
						"""
			self.fitnesses += [finalLoss]
			print("Member acheived fitness : ", finalLoss)
			member.close()

	def generation(self):
		
		zippedFitnesses = sorted(zip(self.fitnesses, self.chromosomes), key=lambda x: x[0])
		currentTotalFitness = 0.0
		for f, c in zippedFitnesses:
			currentTotalFitness += f

		self.evolutionPlot += [currentTotalFitness]
		print("Current total Error : ", currentTotalFitness)

		bestFitness, bestChromosome = zippedFitnesses[0]

		#newPopulation = [gn.GeneticNetwork(bestChromosome, self.dimensionsForStructure(bestChromosome), self.validActivationFunctions)]
		newChromosomes = [list(bestChromosome)]


		for tournamentNumber in range(self.populationSize - 1):
			m1, m2 = self.tournamentSelect(zippedFitnesses)
			child = self.crossover(m1, m2)
			newChromosomes += [child]
			#dims = self.dimensionsForStructure(child)
			#member = gn.GeneticNetwork(child, dims, self.validActivationFunctions)
			#newPopulation += [member]

		#self.population = newPopulation
		self.chromosomes = newChromosomes

		for c in self.chromosomes:
			print(c)

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
