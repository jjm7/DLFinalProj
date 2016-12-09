import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from datetime import datetime


def load_sample_preprocessed(mode = 'train', train_ratio=0.80, choice='random', min_months=6):
	"""Generator to load train or test features for a business. Preprocesses data by imputing and scaling"""
	data_dir = 'data_1/'
	business_id_list = np.load('business_list.npy')
	#get columns in feature mat
	with open('columns.csv','r') as f:
		cols = f.readline().strip().split(',')
	# get all non one hot encoding columns
	cols_to_scale = [name for name in cols if 'one_hot' not in name] 
	business_id_gen = get_business_id(business_id_list, choice=choice)
	for business_id in business_id_gen:
		#choose business and load their features. TODO: for test mode, go through businesses sequentially
		features = np.load(data_dir + business_id + '_features.npy')
		labels = np.load(data_dir + business_id + '_labels.npy')
		#train test split
		train_size = int(len(features) * train_ratio)
		if train_size < min_months: #less than 6 months of training data
			continue
		train_X, train_Y = features[:train_size, :], labels[:train_size]
		test_X, test_Y = features[train_size:, :], labels[train_size:]
		#impute columns with mean
		custom_impute(train_X)
		imputer = Imputer(strategy='mean')
		train_X = imputer.fit_transform(train_X)
		### scale only appropriate columns. Scaling each individual business messes up differences between businesses. Need to fit on all training data
		# scaler = StandardScaler()
		# df_train_temp = pd.DataFrame(train_X, columns=cols)
		# df_train_temp[cols_to_scale] = scaler.fit_transform(df_train_temp[cols_to_scale])
		# train_X = df_train_temp.values
		if mode=='train':
			yield train_X, train_Y
		else:
			#impute and scale test data using train imputer and scaler, yield results
			custom_impute(test_X)
			test_X = imputer.transform(test_X)
			# df_test_temp = pd.DataFrame(test_X, columns=cols)
			# df_test_temp[cols_to_scale] = scaler.transform(df_test_temp[cols_to_scale])
			# test_X = df_test_temp.values
			if mode=='test':
				yield test_X, test_Y
			if mode=='both':
				yield train_X, train_Y, test_X, test_Y
	return

def get_business_id(business_id_list, choice='random'):
	if choice=='random':
		while True:
			yield random.choice(business_id_list)
	elif choice=='sequential':
		for business_id in business_id_list:
			yield business_id

def custom_impute(X):
	if X[:,2][0]==None: #if Price range is None
		X[:,2] = -1  #set to -1

def load_batch_same_len(batch_size=32, mode='train', train_ratio=0.80, choice='random', min_months=12):
	loader = load_sample_preprocessed(mode = mode, train_ratio=train_ratio, choice=choice, min_months=min_months)
	X_batch, y_batch = loader.next()
	X_batch, y_batch = reshape_and_window(X_batch, y_batch, min_months)
	for train_X, train_Y in loader:
		X_train, y_train = reshape_and_window(train_X, train_Y, min_months)
		X_batch = np.vstack((X_batch, X_train))
		y_batch = np.vstack((y_batch, y_train))
		if X_batch.shape[0]==batch_size:
			yield X_batch, y_batch

def load_batch_all_in_memory(batch_size=32, mode='train', train_ratio=0.80, choice='random', min_months=24):
	#first load all businesses in sequentially
	loader = load_sample_preprocessed(mode = mode, train_ratio=train_ratio, choice='sequential', min_months=min_months)
	X_all = []
	y_all = []
	for train_X, train_Y in loader:
		X_train, y_train = reshape_and_window(train_X, train_Y, min_months)
		X_all.append(X_train)
		y_all.append(y_train)
	print len(X_all), len(y_all)

def reshape_and_window(X, y, min_months):
	X, y = reshape_inputs(X, y)
	return X[:, (X.shape[1]-min_months):, :], y[:, (y.shape[1]-min_months):, :]

def reshape_inputs(X, y):
	return np.reshape(X, (1, X.shape[0], X.shape[1])), np.reshape(y, (1, y.shape[0], 1))

if __name__ == '__main__':
	start = datetime.now()

	load_batch_all_in_memory()
	# load_batch_same_len(choice='sequential', min_months=12)
	# for i in xrange(4):
	# 	X, y = load_sample_preprocessed(mode='test').next()
	# 	print X.shape, y.shape

	end = datetime.now()
	print "this took: ", end - start
