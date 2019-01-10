from nltk.stem import PorterStemmer
import re
import os
import json
import numpy as np
from numpy import linalg as LA
import math

df = {} # {term : freq}
N = 0

def preprocessing(doc):
	doc = re.sub('\n', '', doc)
	sentences = re.split('\.', doc)
	sentences = list(filter(None, sentences))
	originalSent = sentences
	sentList = []
	wordsList = []
	stop = open('english.stop', 'r').read().split()  # Load Stopword
	stemmer = PorterStemmer()  # Using Porter’s algorithm
	for i in range(len(sentences)):
		string = sentences[i].lower()
		content = re.split(' ',re.sub('[^a-z]+', ' ', string))
		content = list(filter(None, content))
		words = [k for k in content if k not in stop]  # Stopword removal
		words = [stemmer.stem(word) for word in words]  # Stemming
		wordsList.extend(words)
		sentList.append(words)
	return wordsList, sentList, originalSent #list of list[word]

def makeDict(docs):
	dictionary = ["NULLTERM"] # let dictionary[0] = "NULLTERM"
	df["NULLTERM"] = 0
	for doc in docs:
		dictionary.extend(doc['wordsList'])
	dictionary = sorted(list(set(dictionary)))
	for term in dictionary:
		df[term] = 0
		for doc in docs:
			if term in doc['wordsList']:
				df[term] += 1
	with open("dictionary.txt", "w") as f:  # Save the dictionary
		f.write('{:<10}{:<20}{:<20}\n'.format('t_index', 'term', 'df'))
		for index in range(1, len(dictionary)): #index = 0 is NULLTERM
			f.write('{:<10}{:<20}{:<20}\n'.format(index, dictionary[index], df[dictionary[index]]))
	return dictionary

def tfidf(doc, dictionary):
	docID = doc['docID']
	wordsList = doc['wordsList']
	tf = {} #{term : count}	
	for term in wordsList:
		if term in tf:
			tf[term] += 1
		else:
			tf[term] = 1
	terms = sorted(list(set(wordsList))) #消除重複值並排序
	# tfidf = {} #{term : tfidf}	
	# for term in terms:
	# 	idf = math.log(N / df[term])
	# 	tfidf[term] = tf[term] * idf
	# unit = LA.norm(list(tfidf.values()))

	with open('tfidf_output/' + str(docID) +".txt", "w") as f: #Save file in 'tfidf_output' folder
		f.write('termCount: {:<10}\n'.format(len(terms)))
		for term in terms:
			# tfidf[term] /= unit
			idf = math.log(N / df[term])
			tfidf = tf[term] * idf
			f.write('{:<20}{:<20}\n'.format(term, tfidf))

def readVec(doc): #讀取txt檔案
	dic = {} # {index : tfidf}
	docList = re.split('\n', doc)
	for i in range(1, len(docList)):
		items = re.split('\s', docList[i])
		items = list(filter(None, items))
		if(len(items) == 2):
			dic[items[0]] = float(items[1])
	return dic #return a dictionary key-value pair{index : tfidf}

def sentTfidf(doc, dic):
	print('docID:', doc['docID'])
	print('title:', doc['title'])
	sentList = doc['sentList']
	originalSent = doc['originalSent']
	sumTfidf = []
	for sent in sentList:
		sumTfidf.append(sum(dic[word] for word in sent))
	return originalSent[np.argmax(sumTfidf)]

if __name__ == '__main__':
	#建立字典
	documents = [None]
	for filename in os.listdir(os.getcwd() + "/data"):
		if not filename.startswith('.'):
			N += 1
			with open("data/"+filename, 'r', errors='ignore') as f:
				file = f.read()
				text = json.loads(file) # dict type
				content = text['content']
				wordsList, sentList, originalSent = preprocessing(content)
				text['docID'] = N
				text['wordsList'] = wordsList
				text['sentList'] = sentList
				text['originalSent'] = originalSent
				documents.append(text)
				
	print('Number of docs: ',N)
	dictionary = makeDict(documents[1:])
	print("Dictionary made.")

	#計算TF-IDF
	for doc in documents[1:]:
		tfidf(doc, dictionary)
	print("tf-idf calculated.")

	for docID in range(1, N+1):
		with open("tfidf_output/"+ str(docID) + '.txt', 'r', errors='ignore') as f:
			tfidfDic = readVec(f.read())
			print(sentTfidf(documents[docID], tfidfDic))
