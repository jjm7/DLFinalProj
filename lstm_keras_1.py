import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential, model_from_json
from keras.layers import TimeDistributed, Dense, Activation, LSTM
from keras.callbacks import TensorBoard
from sklearn.metrics import mean_squared_error
from datetime import datetime
import data_loader
import heapq
import dataset
from collections import defaultdict
import dataset
import cPickle as pickle

def main(load_model=False):
	#set parameters
	num_units = 200
	num_mini_batches = 100
	nb_epoch = 10
	batch_size = 32
	activation_hist_freq = int(nb_epoch/10)

	with open('columns.csv','r') as f:
		cols = f.readline().strip().split(',')
	input_dim = len(cols)


	# train_loader = data_loader.load_batch_preprocessed(mode='train', train_ratio=0.80, choice='sequential') 
	# train_loader = data_loader.load_batch_same_len(batch_size=batch_size, choice='sequential', min_months=min_months)
	train_X, train_Y = data_loader.load_batch_all_in_memory(batch_size=batch_size, train_ratio=train_ratio, min_months=min_months)

	# create and fit the LSTM network
	if load_model:
		model = load_model_from_file()
	else:
		model = Sequential()
		model.add(LSTM(num_units, input_dim=input_dim, return_sequences=True))
		model.add(LSTM(num_units, return_sequences=True))
		model.add(TimeDistributed(Dense(1))) #with no activation specified, linear is implied
		model.compile(loss='mean_squared_error', optimizer='adam')
	#print model structure and # of weights
	model.summary()
	#add callbacks. One for tensorboard visualization. Should be able to visualize activation histograms over time
	callbacks = [TensorBoard(log_dir='./logs', histogram_freq=activation_hist_freq, write_graph=True, write_images=False)]

	print "training model"
	# for i in xrange(num_mini_batches):
	# 	train_X, train_Y = train_loader.next()
	# 	# reshape input to be [samples, time steps, features]
	# 	# train_X, train_Y = reshape_inputs(train_X, train_Y)
	# 	model.fit(train_X, train_Y, nb_epoch=nb_epoch, batch_size=batch_size, verbose=0)
	# 	loss = model.evaluate(train_X, train_Y, verbose=0)
	# 	print 'business #', i, ' loss:', loss

	model.fit(train_X, train_Y, nb_epoch=nb_epoch, batch_size=batch_size, verbose=2, callbacks=callbacks)
	loss = model.evaluate(train_X, train_Y, verbose=0)
	print "overall loss", loss

	#save model and weights
	with open(model_structure_name, 'wb') as f:
		f.write(model.to_json())
	model.save_weights(model_weights_name)

	return

def evaluate_example(business_id, model = None):
	if model is None:
		model = load_model_from_file()
	
	train_loader = data_loader.load_sample_preprocessed(mode='test', train_ratio=train_ratio, choice='sequential', min_months=min_months, business_id_list=[business_id])
	features, labels, train_size, _ = train_loader.next()

	results = evaluate_model_rmse_on_business(model, features, labels, train_size)
	features = results['features']
	labels = results['labels']
	predicted = results['predicted']
	print "train rmse: ", results['train_rmse']  # Printing RMSE 
	print "test rmse: ", results['test_rmse']  # Printing RMSE 

	#print information about the business
	db = dataset.connect('sqlite:///yelp.db')
	result = db.query("""
        SELECT * 
        FROM business_info
        WHERE business_id='%s'
        """%business_id)
	print result.next()
	result = db.query("""
        SELECT * 
        FROM business_categories 
        WHERE business_id='%s'
        """%business_id)
	print 'categories:', [row['category'] for row in result]

	#plot training and testing predictions along with ground truth
	plt.figure(1)
	plt.plot(range(features.shape[0]), labels,'b')
	plt.plot(range(min_months), predicted[:min_months],'g')
	plt.plot(range(min_months, features.shape[0]), predicted[min_months:],'r')
	plt.title('Example and lstm prediction')
	plt.xlabel('# months')
	plt.ylabel('# of reviews')
	plt.show()


def find_top_n_examples(num, model = None, mode='test'):
	"""find top and bottom n examples based on train or test rmse. This takes around 15 minutes so use sparingly"""
	if model is None:
		model = load_model_from_file()
	train_loader = data_loader.load_sample_preprocessed(mode='test', train_ratio=train_ratio, choice='sequential', min_months=min_months)
	all_rmse = []
	all_relative_err = []
	cat_errors = Category_Errors()
	i = 1
	for features, labels, train_size, business_id in train_loader:
		#start from min_months before train_size, since that's when training started
		results = evaluate_model_rmse_on_business(model, features, labels, train_size)
		#do we want top/bottom test or train rmse
		if mode=='test':
			rmse = results['test_rmse']
			relative_err = results['test_relative_err']
			cat_errors.add_cat_error(business_id, rmse, relative_err)
		elif mode=='train':
			rmse = results['train']
		all_rmse.append((rmse, business_id))
		all_relative_err.append((relative_err, business_id))
		if i%1000==0:
			print i
		i+=1
	# write test rmse right here. just take average  of all_rmse
	rmse_vals, _ = zip(*all_rmse) #unzip list of tuples to get list of rmse values only
	print mode + ' rmse for all buzi', sum(rmse_vals)/float(len(rmse_vals))
	rel_err_vals, _ = zip(*all_relative_err)
	print mode + ' relative error for all business', np.nanmean(rel_err_vals)
	print heapq.nlargest(num, all_rmse, key = lambda x: x[0])
	print heapq.nsmallest(num, all_rmse, key = lambda x: x[0])
	#save dictionary of category errors
	cat_errors.save_category_errors_dict('business_category')

class Category_Errors():
	def __init__(self):
		self.category_rmse = defaultdict(list)
		self.category_rel_err = defaultdict(list)

	def add_cat_error(self, business_id, rmse, relative_err):
		cats = self.get_business_categories(business_id)
		for category in cats:
			self.category_rmse[category].append(rmse)
			self.category_rel_err[category].append(relative_err)

	def save_category_errors_dict(self, file_name):
		with open(file_name+"_rmse", "wb") as f:
			pickle.dump(self.category_rmse, f)
		with open(file_name+"_rel_err", "wb") as f:
			pickle.dump(self.category_rel_err, f)

	def get_business_categories(self, business_id):
		result = db.query("""
		    SELECT * 
		    FROM business_categories 
		    WHERE business_id='%s'
		    """%business_id)
		return [row['category'] for row in result]

def evaluate_model_rmse_on_business(model, features, labels, train_size):
	"""Evaluate a model's performance on a business given its features, labels, and training set size
	Assumes training set was min_months before train_size, which is true for our standardized time step training"""
	features = features[(train_size-min_months):, :]
	labels = labels[(train_size-min_months):]
	all_X, all_Y = reshape_inputs(features, labels)
	predicted = np.squeeze(model.predict(all_X))
	train_rmse = compute_rmse(predicted[:min_months], labels[:min_months])
	test_rmse = compute_rmse(predicted[min_months:], labels[min_months:])
	test_relative_err = compute_relative_error(predicted[min_months:], labels[min_months:])
	#can compute relative error here too
	return {'train_rmse':train_rmse, 'test_rmse':test_rmse, 'predicted':predicted,'test_relative_err':test_relative_err ,'features':features, 'labels':labels}


def compute_rmse(predicted, labels):
	return math.sqrt(mean_squared_error(predicted, labels))

def compute_relative_error(predicted, labels):
	labels= [x if x != 0  else 0.1 for x in labels]
	return np.mean(np.abs(labels - predicted)/labels)


def reshape_inputs(X, y):
	return np.reshape(X, (1, X.shape[0], X.shape[1])), np.reshape(y, (1, y.shape[0], 1))


def load_model_from_file():
	with open(model_structure_name, 'rb') as f:
		json_string = f.readline()
	model = model_from_json(json_string)
	model.load_weights(model_weights_name, by_name=False)
	return model

if __name__ == '__main__':
	start = datetime.now()

	model_structure_name = 'lstm_2l_200hu_LSTM_24month_pt8Train_MAEloss_100epoch.json'
	model_weights_name = 'lstm_weights_2l_200hu_LSTM_24month_pt8Train_MAEloss_100epoch'
	min_months = 24
	train_ratio = 0.8
	db = dataset.connect('sqlite:///yelp.db')

	# main()
	# find_top_n_examples(6, model = None, mode='test')
	evaluate_example('sIyHTizqAiGu12XMLX3N3g')

 	end = datetime.now()
	print "this took: ", end - start

