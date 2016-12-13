import matplotlib.pyplot as plt
import numpy as np

def make_cum_labels():
	data_dir = 'data_1/'
	business_id_list = np.load('business_list.npy')
	#get columns in feature mat
	for business_id in business_id_list:
		labels = np.load(data_dir + business_id + '_labels.npy')
		cum_labels = []
		total_count = 0
		for num in labels:
			total_count += num
			cum_labels.append(total_count)
		np.save(data_dir + business_id + '_cum_labels', np.array(cum_labels))

def hstack_glove_vecs():
	data_dir = 'data_1/'
	business_id_list = np.load('business_list.npy')
	#get columns in feature mat
	i=1
	for business_id in business_id_list:
		features = np.load(data_dir + business_id + '_features.npy')
		glove_vec = np.load('glove_vecs/'+business_id+'_catvec.npy')
		#repeat glove vec as many times as # of rows in features
		glove_mat = np.tile(glove_vec,(features.shape[0],1))
		new_features = np.hstack((features, glove_mat))
		np.save(data_dir + business_id + '_features.npy', new_features)
		if i%1000==0:
			print i 
		i+=1
	
if __name__ == '__main__':
	hstack_glove_vecs()