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

def main(load_model=False):
	#set parameters
	num_units = 200
	num_mini_batches = 100
	nb_epoch = 100
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
	i = 1
	for features, labels, train_size, business_id in train_loader:
		#start from min_months before train_size, since that's when training started
		results = evaluate_model_rmse_on_business(model, features, labels, train_size)
		#do we want top/bottom test or train rmse
		if mode=='test':
			rmse = results['test_rmse']
		elif mode=='train':
			rmse = results['train']
		all_rmse.append((rmse, business_id))
		i+=1
	print heapq.nlargest(num, all_rmse, key = lambda x: x[0])
	print heapq.nsmallest(num, all_rmse, key = lambda x: x[0])


def evaluate_model_rmse_on_business(model, features, labels, train_size):
	"""Evaluate a model's performance on a business given its features, labels, and training set size
	Assumes training set was min_months before train_size, which is true for our standardized time step training"""
	features = features[(train_size-min_months):, :]
	labels = labels[(train_size-min_months):]
	all_X, all_Y = reshape_inputs(features, labels)
	predicted = np.squeeze(model.predict(all_X))
	train_rmse = compute_rmse(predicted[:min_months], labels[:min_months])
	test_rmse = compute_rmse(predicted[min_months:], labels[min_months:])
	#can compute relative error here too
	return {'train_rmse':train_rmse, 'test_rmse':test_rmse, 'predicted':predicted, 'features':features, 'labels':labels}


def compute_rmse(predicted, labels):
	return math.sqrt(mean_squared_error(predicted, labels))

def compute_relative_error(predicted, labels):
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

	model_structure_name = 'lstm_structure_3.json'
	model_weights_name = 'lstm_weights_3'
	min_months = 24
	train_ratio = 0.8

	main()
	# find_top_n_examples(6, model = None, mode='test')
	# evaluate_example('2lF_8RZRBd9STfk_QwAtmg')

 	end = datetime.now()
	print "this took: ", end - start

