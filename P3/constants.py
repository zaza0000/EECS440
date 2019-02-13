DATA_PATH = './volcanoes/volcanoes'
ENABLE_VAL = 0			#If 0, use cross validation. If 1, use full sample.
ALGORITHM = 2			#1=dtree, 2=nbayes, 3=logreg
P = 0					#Flip probability for writeup 3 and 4
ITER = 10				#Number of iterations

NUM_BAG = 10             #classifier num for bagging

#Decision tree constants
MAX_DEPTH = 1
ENABLE_GAIN = 0  		#If 0, use information gain. If 1, use gain ratio.
EPS = 1.0e-10

#Bayes constants
NUM_BINS = 5			#Number of bins for any continuous feature
M = 10  					#Value of m for the m-estimate. If negative, use Laplace smoothing.

#LR constants
LAMBDA = 1			#The constant λ that times the penalty term
LR = 0.01				#Learning rate
ITER2 = 100			#Number of BPTT per iteration