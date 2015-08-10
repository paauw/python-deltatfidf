from __future__ import division
import numpy as np
import math

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# c1 = [ 'dit is de eerste text', 'de is de tweede text nog wat info', 'dit is de derde text wat grappig dit henk'];
# c2 = [ 'dit is de tweede class text', 'jaja wat een uitstekende test', 'jaja dit is lastig zeg'];

# print start[1].split().count('de')

class deltaTFIDF:
	def __init__(self, class1, class2, max_features=0, treshold=0.1):
		self.treshold = treshold
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

			# We throw out anything below the treshold, because the really low
			# log values mean that there is almost no difference between classes
		# 	if self.cc_dict[term]['log'] < self.treshold:
		# 		self.below_thres.append(term)

		# self.features = [a for a in self.features if a not in self.below_thres]

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

#delta = deltaTFIDF(c1,c2, max_features=5)
# training = delta.vectorize(c1 + c2)

# model = RandomForestClassifier(n_estimators = 100)
# #model = SVC(probability=1)
# model.fit( training, [1,1,1,0,0,0])

# print sorted(delta.features, key=lambda x:delta.cc_dict[x])
# print delta.vector('henk henk')
# print model.predict_proba( delta.vector('henk') ) 
