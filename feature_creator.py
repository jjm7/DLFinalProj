import sqlite3
import dataset
from datetime import datetime
import collections
import numpy as np
from fixNames import vectorizeList #uncomment
import random

def write_features_to_csv():
	data_dir = 'data_1/'
	businesses = get_valid_businesses()
	business_ids = [] # to hold list of business_ids
	j=1
	for row in businesses:
		#TODO: get categories, and feed into create_features
		business_ids.append(row['business_id'])
		features, labels = create_features_for_business(row)
		print features.shape, label.shape
		print features
		break
		if len(features)<2:
			continue
		np.save(data_dir + row['business_id'] + '_features', np.array(features))
		np.save(data_dir + row['business_id'] + '_labels', np.array(labels))
		if j%1000==0:
			print j
		j+=1
	#save list of business_ids
	np.save('business_list', np.array(business_ids))
	return

def get_valid_businesses():
	result = db.query("""
        SELECT *
        FROM business_info 
        WHERE review_count>30
        """)
	return result

def get_business_categories(business_id):
	result = db.query("""
        SELECT * 
        FROM business_categories 
        WHERE business_id='%s'
        """%business_id)
	return [row['category'] for row in result]

def write_cols():
	with open('columns.csv', 'wb') as csv_file:
		#static feature names
		cols = ['latitude', 'longitude', 'price_range']
		cols.extend(['one_hot_'+state for state in all_states.keys()])
		cols.extend(['categories_vec_'+str(i) for i in xrange(1,26)])
		#dynamic feature names
		cols.extend(['month', 'months_since_first_review', 'review_count', 'total_review_count', 'avg_stars_for_month', 'cum_rating', \
				'votes_funny', 'votes_cool', 'votes_useful', 'tip_count', 'total_tip_count'])

		csv_file.write(','.join(cols)+'\n')


def create_features_for_business(business_row):
	business_id = business_row['business_id']
	#create static features
	static_features = [business_row['latitude'],business_row['longitude'],business_row['price_range']]
	static_features.extend(state_to_one_hot(business_row['state']))
	static_features.extend( vectorizeList(get_business_categories(business_id)) ) #uncomment
	#create dynamic features
	dynamic_features_results = db.query(dynamic_features_sql%business_id)
	feature_adder = Dynmamic_Features()
	features = []
	labels = []
	#add first row
	row = dynamic_features_results.next()
	prev_date = datetime(row['year'], row['month'], 1)
	vec = list(static_features)
	vec.extend(feature_adder.add_dynamic_features_real_row(row))
	features.append(vec)
	#add remaining rows in result set and interpolate months
	for row in dynamic_features_results:
		next_date = datetime(row['year'], row['month'], 1)
		curr_date = add_month(prev_date)
		#interpolate dates if there are month gaps between rows
		while curr_date < next_date:
			vec = list(static_features)
			vec.extend(feature_adder.add_dynamic_features_interpolated_row(curr_date))
			features.append(vec)
			labels.append(0)
			prev_date = curr_date
			curr_date = add_month(curr_date)
		#add next row after interpolation
		prev_date = next_date
		vec = list(static_features)
		vec.extend(feature_adder.add_dynamic_features_real_row(row))
		features.append(vec)
		labels.append(row['review_count'])
	# labels.append(-1)
	features.pop() #remove last row with no label
	return (features, labels)

def add_month(a_date):
    if a_date.month==12:
        return datetime(a_date.year+1, 1, 1)
    else:
        return datetime(a_date.year, a_date.month+1, 1)

class Dynmamic_Features:
	def __init__(self):
		self.month_num = -1
		self.prev_month = 0
		self.prev_avg_stars = 0
		self.total_review_count = 0
		self.total_stars = 0
		self.total_tips = 0
		self.current_rating = None

	def add_dynamic_features_real_row(self, result_row):
		self.month_num += 1
		self.prev_month = result_row['month']
		self.prev_avg_stars = result_row['avg_stars'] if result_row['avg_stars'] is not None else self.prev_avg_stars
		if result_row['review_count']>0:
			self.total_stars += result_row['avg_stars']*result_row['review_count']
		self.total_review_count += result_row['review_count']
		self.total_tips += result_row['tip_count']
		if self.total_review_count>0:
			self.current_rating = 1.0*self.total_stars/self.total_review_count
		return [result_row['month'], self.month_num, result_row['review_count'], self.total_review_count, self.prev_avg_stars, \
			self.current_rating, result_row['votes_funny'], result_row['votes_cool'], result_row['votes_useful'], \
			result_row['tip_count'], self.total_tips]

	def add_dynamic_features_interpolated_row(self, date):
		self.month_num += 1
		return [date.month, self.month_num, 0, self.total_review_count, self.prev_avg_stars, \
			self.current_rating, 0, 0, 0, \
			0, self.total_tips]


def state_to_one_hot(state):
	vec = [0]*len(all_states)
	vec[all_states[state]] = 1
	return vec

def make_states_dict():
	#make look-up dict for states for one-hot encoding later
	state_results = db.query("""
	        SELECT DISTINCT state
			FROM business_info
			ORDER BY state
	        """)
	#maps state string to index
	all_states = collections.OrderedDict()
	i=0
	for row in state_results:
		all_states[row['state']] = i
		i+=1
	return all_states

if __name__ == '__main__':
	start = datetime.now()
	# connecting to a SQLite database
	db = dataset.connect('sqlite:///yelp.db')
	
	all_states = make_states_dict()

	dynamic_features_sql ="""
	WITH businesses AS (
	    SELECT *
	    FROM business_info
	    WHERE business_id ='%s'
	), business_review AS( --review stats per year-month
	    SELECT businesses.business_id, year, month, COUNT() AS review_count, AVG(review.stars) AS avg_stars,
	              SUM(votes_funny) AS votes_funny, SUM(votes_cool) AS votes_cool, SUM(votes_useful) AS votes_useful
	    FROM businesses
	    INNER JOIN review ON businesses.business_id=review.business_id
	    GROUP BY businesses.business_id, year, month
	), business_tip AS( --tip count per year-month
	    SELECT businesses.business_id, year, month, COUNT() AS tip_count
	    FROM businesses
	    INNER JOIN tip ON businesses.business_id=tip.business_id
	    GROUP BY businesses.business_id, year, month
	), combined AS (--Use two left joins to do full outer join on business_id, year, month
	    SELECT business_review.*, business_tip.business_id AS business_id_2, business_tip.year AS year_2, 
	            business_tip.month AS month_2, business_tip.tip_count
	    FROM business_review 
	    LEFT JOIN business_tip 
	        ON business_review.business_id = business_tip.business_id
	        AND business_review.year = business_tip.year
	        AND business_review.month = business_tip.month
	    UNION ALL
	    SELECT business_review.*, business_tip.*
	    FROM business_tip
	         LEFT JOIN business_review
	            ON business_review.business_id = business_tip.business_id
	            AND business_review.year = business_tip.year
	            AND business_review.month = business_tip.month
	    WHERE business_review.business_id IS NULL
	) --finally construct table with both reviews and tips. Combine business_id, year, and month columns into one
	SELECT CASE WHEN business_id IS NULL THEN business_id_2 ELSE business_id END AS business_id,
	        CASE WHEN business_id IS NULL THEN year_2 ELSE year END AS year,
	        CASE WHEN business_id IS NULL THEN month_2 ELSE month END AS month,
	        COALESCE(review_count, 0) AS review_count, avg_stars, COALESCE(votes_funny, 0) AS votes_funny,
	        COALESCE(votes_cool, 0) AS votes_cool, COALESCE(votes_useful, 0) AS votes_useful,
	        COALESCE(tip_count, 0) AS tip_count
	FROM combined
	ORDER BY year, month
	"""

	# write_cols()

	write_features_to_csv()

	end = datetime.now()
	print "this took: ", end - start