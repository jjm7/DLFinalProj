import string
def fixNames(myString):
	temp = string.replace(myString,'_',' ')
	temp = string.replace(temp,'(','')
	temp = string.replace(temp,')','')
	temp = string.replace(temp,'&','')
	return temp
