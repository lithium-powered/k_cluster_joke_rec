import random
import numpy as np
import math
import csv
from scipy import io
import scipy
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors

#load data
data1 = io.loadmat('joke_data/joke_train.mat')
validation = np.loadtxt("joke_data/validation.txt", delimiter=',')

trainingData = data1['train']

validationUsers = validation[:,0].astype(int)
validationJokes = validation[:,1].astype(int)
validationScoring = validation[:,2]

#2.2 simplest
'''
#simplest recommender system
avgJokeRating = np.nanmean(trainingData, axis=0)
pred = np.fromfunction(lambda i: avgJokeRating[validationJokes[i]-1] > 0, (validationJokes.size,), dtype=int)
print metrics.accuracy_score(pred,validationScoring)
'''

#2.2 more advanced
'''
#more advanced recommender system
trainingDataNoNan = np.nan_to_num(trainingData)
numNeighbors = 1000
neigh = NearestNeighbors(n_neighbors=numNeighbors+1).fit(trainingDataNoNan)

pred = []
for pair in validation:
	distances, indices = neigh.kneighbors(trainingDataNoNan[pair[0]-1])
	indices = indices[0][1:,]
	if np.average(trainingDataNoNan[indices],axis=0)[pair[1]-1] > 0:
		pred.append(1)
	else:
		pred.append(0)
print metrics.accuracy_score(pred,validationScoring)
'''

####2.3.2
'''
d=20
trainingDataNoNan = np.nan_to_num(trainingData)
userLatent, sLatent, jokeLatent = scipy.sparse.linalg.svds(trainingDataNoNan,k=d )
jokeLatent = jokeLatent.T
Us = userLatent*sLatent
MSE = float(0)
for i in range(0,np.size(Us, axis = 0)):
	for j in range(0, np.size(jokeLatent, axis = 0)):
		if (not math.isnan(trainingData[i][j])):
			MSE += math.pow(np.dot(Us[i],jokeLatent[j]) - trainingData[i][j],2)

print MSE

ratingPred = np.matrix(userLatent)*np.diag(sLatent)*np.matrix(jokeLatent).T
pred = []
for pair in validation:
	if ratingPred.item((pair[0]-1,pair[1]-1)) > 0:
		pred.append(1)
	else:
		pred.append(0)
print metrics.accuracy_score(pred,validationScoring)
'''

####2.3.3 + Kaggle
d = 20
normRatings = trainingData/float(10)
u = 2*np.random.rand(np.size(normRatings, axis=0),d)-1
v = 2*np.random.rand(np.size(normRatings, axis=1),d)-1

iterations = 10
for k in range(0,iterations):
	for i in range(0,np.size(u, axis=0)):
		summation = np.zeros(d)
		for j in range(0,np.size(v, axis=0)):
			if (not math.isnan(trainingData[i][j])):
				summation = summation + v[j].T*normRatings.item(i,j)
		u[i] = (np.linalg.inv(np.dot(v.T,v)+np.diag(15*np.ones(d)))*np.matrix(summation).T).flatten()
	for j in range(0,np.size(v, axis=0)):
		summation = np.zeros(d)
		for i in range(0, np.size(u, axis=0)):
			if (not math.isnan(trainingData[i][j])):
				summation = summation + u[i].T*normRatings.item(i,j)
		v[j] = (np.linalg.inv(np.dot(u.T,u)+np.diag(15*np.ones(d)))*np.matrix(summation).T).flatten()

MSE = float(0)
for i in range(0,np.size(u, axis = 0)):
	for j in range(0, np.size(v, axis = 0)):
		if (not math.isnan(trainingData[i][j])):
			MSE += math.pow(np.dot(u[i],v[j]) - normRatings[i][j],2)

print MSE

ratingPred = np.matrix(u)*np.matrix(v).T
pred = []
for pair in validation:
	if ratingPred.item((pair[0]-1,pair[1]-1)) > 0:
		pred.append(1)
	else:
		pred.append(0)
print metrics.accuracy_score(pred,validationScoring)

kaggle = np.loadtxt("joke_data/query.txt", delimiter=',')

pred = []
for pair in kaggle:
	if ratingPred.item((pair[1]-1,pair[2]-1)) > 0:
		pred.append(1)
	else:
		pred.append(0)

spamCSV = csv.writer(open('kagglePredictions.csv', 'wt'))
spamCSV.writerow(['Id', 'Category'])
for i in range(0,len(kaggle)):
	spamCSV.writerow([i+1,int(pred[i])])

