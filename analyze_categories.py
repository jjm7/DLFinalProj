from collections import defaultdict
import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np


def get_sorted_cat_errors(file_name):
	"""load category errors dictionary and print out categories sorted by error"""
	cat_err = load_dict(file_name)
	#median or mean error for categories with at least n values. 
	mean_errs = [(np.median(vals), cat) for cat, vals, in cat_err.iteritems() if len(vals)>10]
	print 'number of categories', len(mean_errs)
	print sorted(mean_errs)

def show_errors_for_cat(file_name, category):
	"""prints out list of errors for given category and plots histogram of errors"""
	cat_err = load_dict(file_name)
	errors = cat_err[category]
	print "all errors:", errors
	plt.figure(1)
	plt.hist(errors)
	plt.title("histogram of errors for category "+category)
	plt.xlabel('error')
	plt.show()


def load_dict(file_name):
	with open(file_name,'rb') as f:
		cat_err = pickle.load(f)
	return cat_err

if __name__ == '__main__':
	file_name = 'business_category_rel_err'
	# get_sorted_cat_errors(file_name)
	show_errors_for_cat(file_name, 'German')
