import nbayes
import logreg
from math import *

#This file is for writeup 1
#To run, directly run this file, 
#python classifier_comparison.py
bayes_err_rates = nbayes.compute_err_rates()
lg_err_rates = logreg.compute_err_rates()
dif_err_rates = []
avg_bayes_err_rate = 0
avg_lg_err_rate = 0
avg_dif_err_rate = 0
length = len(bayes_err_rates)
for i in range(length):
	dif_err_rate = bayes_err_rates[i] - lg_err_rates[i]
	dif_err_rates.append(dif_err_rate)
	avg_bayes_err_rate += bayes_err_rates[i]
	avg_lg_err_rate += lg_err_rates[i]
	avg_dif_err_rate += dif_err_rate
avg_bayes_err_rate /= length
avg_lg_err_rate /= length
avg_dif_err_rate /= length
s = 0
for i in range(length):
	s += pow(dif_err_rates[i] - avg_dif_err_rate, 2)
s = sqrt(s / (length * (length - 1)))
min = avg_dif_err_rate - 2.776 * s
max = avg_dif_err_rate + 2.776 * s
if(0 >= min and 0 <= max):
	print("The interval is [%.3f, %.3f], the null hypothesis can't be rejected" % (min, max))
else:
	print("The interval is [%.3f, %.3f], the null hypothesis can be rejected" % (min, max))