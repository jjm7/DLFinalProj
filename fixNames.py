import string
import glove as Glove
import numpy as np


def fixNames(myString):
	temp = string.replace(myString,'_',' ')
	temp = string.replace(temp,'(','')
	temp = string.replace(temp,')','')
	temp = string.replace(temp,'&','')
	return temp

def vectorizeList(listMe,model):
	#fix the list and make it into a sentence
	vec = np.zeros(model.no_components)
	allCharacteristics = string.join(listMe)
	sentence = fixNames(allCharacteristics)
	wordList = sentence.split()
	
	#iterator over sentence, vectorize each word and add them up
	for i,dum in enumerate(wordList):
		print wordList[i]
		
		try:
			temp = model.word_vectors[model.dictionary[wordList[i]]]
		#	break
		except:
			print('EXCEPTION' + wordList[i] + ' not in Glove dictionary')
			temp = np.zeros(model.no_components)
		vec = vec + temp

	return vec 


#-------------------------------TEST SCRIPT ---------------------------------------------
#import model
model = Glove.Glove.load_stanford('glove.6B.50d.txt')

#list of restaruance catagories (or city)
restaurantChars = ['brazilian & (steakhouse) fartmonkerzz', 'bananas_warehouse', 'ice bar']
wordVector = vectorizeList(restaurantChars,model)
print wordVector
