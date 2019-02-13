DATA_PATH = './voting/voting'
ALGORITHM = 1
NUM_OF_FEATURES = 1 # greater or equal 1

#Decision tree constants
MAX_DEPTH = 20
ENABLE_GAIN = 0  		#If 0, use information gain. If 1, use gain ratio.
EPS = 1.0e-10

#Bayes constants
NUM_BINS = 5			#Number of bins for any continuous feature
M = 10  					#Value of m for the m-estimate. If negative, use Laplace smoothing.

#LR constants
LAMBDA = 1			#The constant λ that times the penalty term
LR = 0.01				#Learning rate
ITER2 = 30			#Number of BPTT per iteration