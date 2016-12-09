import json
import sqlite3
import dataset
import re
from datetime import datetime

start = datetime.now()

# connecting to a SQLite database
db = dataset.connect('sqlite:///yelp.db')

conn = sqlite3.connect("yelp.db")
conn.isolation_level = None

db.query('PRAGMA synchronous=OFF;')
db.query('PRAGMA journal_mode=TRUNCATE;')

# print db.tables
# print db['business_info'].columns

def insert_business_categories():
	"""There are 1017 unique categories so the table would have 86M elem if we did one-hot
	encoding, just add word to category column instead"""
	filename = "yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json"
	table = db['business_categories']
	with open(filename, 'r') as f:
		i = 1
		chunk = []
		for line in f:
			business = json.loads(line)
			for cat in business['categories']:
				#replace hyphen and space with _ if exists
				row = {'category': re.sub(r'\b-\b|\b \b', '_', cat), 'business_id': business['business_id']}
				chunk.append(row)
				if i%1000==0:
					table.insert_many(chunk, ensure=True)
					chunk = []
				i+=1
				if i%10000==0:
					print i 
		#insert remaining
		table.insert_many(chunk, ensure=True)

def insert_data(json_name, table_name, row_maker):
	filename = "yelp_dataset_challenge_academic_dataset/"+json_name
	table = db[table_name]

	c = conn.cursor()

	with open(filename, 'r') as f:
		i = 1
		chunk = []
		for line in f:
			line_as_dict = json.loads(line)
			row = row_maker(line_as_dict)
			if row is not None:
				chunk.append(row)
			if i%1000==0:
				# table.insert_many(chunk, ensure=True)
				# can update a column instead
				c.execute("begin")
				c.executemany("""UPDATE business_info SET price_range=:price_range WHERE business_id=:business_id """, chunk)
				c.execute("commit")
				chunk = []
			i+=1
			if i%10000==0:
				print i 
		#insert remaining
		# table.insert_many(chunk, ensure=True)
		#Or update remaining
		c.execute("begin")
		c.executemany("""UPDATE business_info SET price_range=:price_range WHERE business_id=:business_id """, chunk)
		c.execute("commit")

def business_info_row_maker(business):
	if 'Price Range' in business['attributes']:
		row = {'business_id': business['business_id'], 'price_range': business['attributes']['Price Range']}
		return row
	return None


def user_row_maker(user):
	return {'user_id':user['user_id'], 'name':user['name'], 'review_count':user['review_count'], 'average_stars':user['average_stars'], \
			'votes_funny':user['votes']['funny'], 'votes_useful':user['votes']['useful'], 'votes_cool':user['votes']['cool'], \
			'yelping_since': user['yelping_since'], 'fans':user['fans']}

def review_row_maker(review):
	"""2,685,066 rows total from file word count"""
	dt = datetime.strptime(review['date'], '%Y-%m-%d')
	return {'business_id':review['business_id'], 'user_id':review['user_id'], 'stars':review['stars'], 'text':review['text'], \
			'date':review['date'], 'votes_funny':review['votes']['funny'], 'votes_cool':review['votes']['cool'], \
			'votes_useful':review['votes']['useful'], 'year': dt.year, 'month': dt.month, 'day': dt.day}

def tip_row_maker(tip):
	"""648,902 rows total from file word count"""
	dt = datetime.strptime(tip['date'], '%Y-%m-%d')
	return {'business_id':tip['business_id'], 'user_id':tip['user_id'], 'text':tip['text'], \
			'date':tip['date'], 'likes':tip['likes'], 'year': dt.year, 'month': dt.month, 'day': dt.day}

def print_json_line():
	filename = "yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json"
	with open(filename, 'r') as f:
		i = 1
		for line in f:
			if i==77000:
				review = json.loads(line)
				for k in review:
					print k, review[k]
				break
			i+=1



# insert_data('yelp_academic_dataset_business.json', 'business_info', business_info_row_maker)
print_json_line()

end = datetime.now()
print "this took: ", end - start


