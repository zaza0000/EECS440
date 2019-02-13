DATA_PATH = './volcanoes/volcanoes'

#Bayes constants
ENABLE_VAL = 0			#If 0, use cross validation. If 1, use full sample.
NUM_BINS = 2			#Number of bins for any continuous feature
M = 0  					#Value of m for the m-estimate. If negative, use Laplace smoothing.

#LR constants
LAMBDA = 0.1			#The constant λ that times the penalty term
ITER = 100				#Learning iteration
LR = 0.01				#Learning rate