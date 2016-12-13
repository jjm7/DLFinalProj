import string
import glove as Glove
import numpy as np

model_name = '../glove.twitter.27B.25d.txt'
model = Glove.Glove.load_stanford(model_name)

def fixNames(myString):
	temp = string.replace(myString,'_',' ')
	temp = string.replace(temp,'/',' ')
	temp = string.replace(temp,'(','')
	temp = string.replace(temp,')','')
	temp = string.replace(temp,'&','')
	temp = string.replace(temp,"'s",'')
	return temp

def vectorizeList(listMe):
	#fix the list and make it into a sentence
	vec = np.zeros(model.no_components)
	allCharacteristics = string.join(listMe)
	sentence = fixNames(allCharacteristics)
	wordList = sentence.split()
	
	#iterator over sentence, vectorize each word and add them up
	for i,dum in enumerate(wordList):
		# print wordList[i]
		
		try:
			temp = model.word_vectors[model.dictionary[wordList[i].lower()]]
		#	break
		except:
			print('EXCEPTION ' + wordList[i].lower() + ' not in Glove dictionary')
			temp = np.zeros(model.no_components)
		vec = vec + temp

	return vec 

if __name__ == '__main__':
	#-------------------------------TEST SCRIPT ---------------------------------------------
	#import model
	model = Glove.Glove.load_stanford(model_name)

	#list of restaruance catagories (or city)
	restaurantChars = ['brazilian & (steakhouse) fartmonkerzz', 'bananas_warehouse', 'ice bar']
	wordVector = vectorizeList(restaurantChars)
	print wordVector
	restaurantChars = ['kobe beef','korean fusion','lasVegas']
	wordVector = vectorizeList(restaurantChars)
	print wordVector
