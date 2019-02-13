import bag
import boost
import base
from math import *

#This file is for writeup 1
#To run, directly run this file, 
#python classifier_comparison.py

bagging_err_rates = bag.compute_err_rates()
boosting_err_rates = boost.compute_err_rates()
base_err_rates = base.compute_err_rates()
dif_err_rates = []
bagging_dif_err_rates = []
boosting_dif_err_rates = []
avg_bagging_err_rate = 0
avg_boosting_err_rate = 0
avg_base_err_rate = 0
avg_bagging_dif_err_rate = 0
avg_boosting_dif_err_rate = 0

length = len(bagging_err_rates)
for i in range(length):
	bagging_dif_err_rate = bagging_err_rates[i] - base_err_rates[i]
	bagging_dif_err_rates.append(bagging_dif_err_rate)
	avg_bagging_err_rate += bagging_err_rates[i]
	
	boosting_dif_err_rate = boosting_err_rates[i] - base_err_rates[i]
	boosting_dif_err_rates.append(boosting_dif_err_rate)
	avg_boosting_err_rate += boosting_err_rates[i]
	
	avg_base_err_rate += base_err_rates[i]
	avg_bagging_dif_err_rate += bagging_dif_err_rate
	avg_boosting_dif_err_rate += boosting_dif_err_rate
avg_bagging_err_rate /= length
avg_base_err_rate /= length
avg_bagging_dif_err_rate /= length
avg_boosting_dif_err_rate /= length
bagging_s = 0
boosting_s = 0
for i in range(length):
	bagging_s += pow(bagging_dif_err_rates[i] - avg_bagging_dif_err_rate, 2)
	boosting_s += pow(boosting_dif_err_rates[i] - avg_bagging_dif_err_rate, 2)
bagging_s = sqrt(bagging_s / (length * (length - 1)))
boosting_s = sqrt(boosting_s / (length * (length - 1)))
bagging_min = avg_bagging_dif_err_rate - 2.776 * bagging_s
boosting_min = avg_boosting_dif_err_rate - 2.776 * boosting_s
bagging_max = avg_bagging_dif_err_rate + 2.776 * bagging_s
boosting_max = avg_boosting_dif_err_rate + 2.776 * boosting_s
if(0 >= bagging_min and 0 <= bagging_max):
	print("For bagging and the base, the interval is [%.3f, %.3f], the null hypothesis can't be rejected" % (bagging_min, bagging_max))
else:
	print("For bagging and the base, the interval is [%.3f, %.3f], the null hypothesis can be rejected" % (bagging_min, bagging_max))
if(0 >= boosting_min and 0 <= boosting_max):
	print("For boosting and the base, the interval is [%.3f, %.3f], the null hypothesis can't be rejected" % (boosting_min, boosting_max))
else:
	print("For boosting and the base, the interval is [%.3f, %.3f], the null hypothesis can be rejected" % (boosting_min, boosting_max))