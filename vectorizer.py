from __future__ import division
import numpy as np
import math

# from sklearn.ensemble import RandomForestClassifier

# c1 = [ 'this is the first text', 'this is the second text and some info', 'this is the third text'];
# c2 = [ 'this is text for the second class', 'just to test', 'repeating test test test'];

class deltaTFIDF:
	def __init__(self, class1, class2, max_features=0):
		self.limit = max_features
		self.classP = class1
		self.classN = class2
		self.corpus = class1 + class2

		self.features = set([item for sublist in [a.split() for a in self.corpus] for item in sublist])

		# Build a cache of word counts per class for speed
		self.classCounts()

	def classCounts(self):
		self.cc_dict = {}
		self.below_thres = []

		for term in self.features:
			self.cc_dict[term] = {}
			self.cc_dict[term]['P'] = reduce(lambda x,y: x+y, [0.1] + [1 for doc in self.classP if term in doc])
			self.cc_dict[term]['N'] = reduce(lambda x,y: x+y, [0.1] + [1 for doc in self.classN if term in doc])
			self.cc_dict[term]['log'] = math.log(self.cc_dict[term]['N']/self.cc_dict[term]['P'],2)

		if self.limit:
			self.features = sorted(self.features, key=lambda x: abs(self.cc_dict[x]['log']), reverse=True)[:self.limit]

	def vector(self, text):
		text = text.split()
		vector = []
		# enumerate the features
		for a in self.features:
			if a in text:
				vector.append( text.count(a) * self.cc_dict[a]['log'] )
			else:
				vector.append( 0 )

		return vector

	def vectorize(self, documents):
		matrix = []
		for text in documents:
			matrix.append( self.vector( text ))
		return matrix

# delta = deltaTFIDF(c1,c2, max_features=5)
# training = delta.vectorize(c1 + c2)

# model = RandomForestClassifier()
# model.fit( training, [1,1,1,0,0,0])

# print model.predict_proba( delta.vector('test') ) 
