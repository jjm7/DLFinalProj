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

make_cum_labels()
