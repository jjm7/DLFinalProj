import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential, model_from_json
from keras.layers import TimeDistributed, Dense, Activation, LSTM
from sklearn.metrics import mean_squared_error
from datetime import datetime
import data_loader

def main(load_model=False):
	#set parameters
	num_units = 200
	num_businesses = 1
	epochs_per_business = 400
	with open('columns.csv','r') as f:
		cols = f.readline().strip().split(',')
	input_dim = len(cols)


	# train_loader = data_loader.load_batch_preprocessed(mode='train', train_ratio=0.80, choice='sequential') 
	train_loader = data_loader.load_batch_same_len(batch_size=32, choice='sequential', min_months=12)
	
	# create and fit the LSTM network
	if load_model:
		model = load_model_from_file()
	else:
		model = Sequential()
		model.add(LSTM(num_units, input_dim=input_dim, return_sequences=True))
		model.add(LSTM(num_units, return_sequences=True))
		model.add(TimeDistributed(Dense(1))) #with no activation specified, linear is implied
		model.compile(loss='mean_squared_error', optimizer='adam')

	model.summary()

	print "training model"
	for i in xrange(num_businesses):
		train_X, train_Y = train_loader.next()
		print train_X.shape, train_Y.shape
		# reshape input to be [samples, time steps, features]
		# train_X, train_Y = reshape_inputs(train_X, train_Y)
		model.fit(train_X, train_Y, nb_epoch=epochs_per_business, batch_size=32, verbose=2)
		loss = model.evaluate(train_X, train_Y, verbose=0)
		print 'business #', i, ' loss:', loss
	#save model and weights
	with open(model_structure_name, 'wb') as f:
		f.write(model.to_json())
	model.save_weights(model_weights_name)

	evaluate_model(model)
	return

def evaluate_model(model = None):
	if model is None:
		model = load_model_from_file()
	
	# train_loader = data_loader.load_batch_preprocessed(mode='both', train_ratio=0.80, choice='sequential')
	train_loader = data_loader.load_batch_same_len(batch_size=32, choice='sequential', min_months=12)
	test_loader = data_loader.load_batch_preprocessed(mode = 'test', train_ratio=0.80, choice='sequential', min_months=12)
	train_X, train_Y = train_loader.next()
	train_X, train_Y = train_X[1], train_Y[1]
	test_X, test_Y = test_loader.next()

	# for i in xrange(100):
	# 	train_X, train_Y, test_X, test_Y = train_loader.next()
	train_X, train_Y = reshape_inputs(train_X, train_Y)
	test_X, test_Y = reshape_inputs(test_X, test_Y)

	print test_X.shape
	train_predict = model.predict(train_X)
	test_predict = model.predict(test_X)
	print "train rmse: ", math.sqrt(mean_squared_error(np.squeeze(train_Y), np.squeeze(train_predict)))  # Printing RMSE 
	print "test rmse: ", math.sqrt(mean_squared_error(np.squeeze(test_Y), np.squeeze(test_predict)))  # Printing RMSE 

	plt.figure(1)
	plt.plot(range(train_X.shape[1]), np.squeeze(train_Y),'b')
	plt.plot(range(train_X.shape[1]), np.squeeze(train_predict),'g')
	plt.title('training example and lstm prediction')
	plt.xlabel('# months since first review/tip')
	plt.ylabel('# of reviews')
	plt.show()

	plt.figure(2)
	plt.plot(range(test_X.shape[1]), np.squeeze(test_Y))
	plt.plot(range(test_X.shape[1]), np.squeeze(test_predict))
	plt.title('test example and lstm prediction')
	plt.xlabel('# months since first review/tip')
	plt.ylabel('# of reviews')
	plt.show()

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

	model_structure_name = 'lstm_structure_test.json'
	model_weights_name = 'lstm_weights_test'
	
	# main()
	evaluate_model()

 	end = datetime.now()
	print "this took: ", end - start

