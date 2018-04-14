#!/usr/bin/env python2.7

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix

# read in data
data = pd.read_csv("ism.csv")
data = data.as_matrix()

X = data[:,:6]
y = data[:,6]

# fit neural network classifier
mlp = MLPClassifier(solver = 'sgd',learning_rate_init=.01)

# fit decision tree classifier
dt = DecisionTreeClassifier(criterion="entropy")

# create 10 stratified samples
skf = StratifiedKFold(n_splits=10)

mlp_errors = []
mlp_Fs = []
dt_errors = []
dt_Fs = []

for train_index, test_index in skf.split(X,y):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
        # train classifiers using training data
	mlp.fit(X_train, y_train)
	dt.fit(X_train, y_train)
	mlp_pred = mlp.predict(X_test)
	dt_pred = dt.predict(X_test)
	
	n_count = 0
	p_count = 0
	for i in mlp_pred:
		if i == 1:
			n_count += 1
		else:
			p_count += 1

	# get confusion matrix = [tn, fp, fn, tp]
	mlp_cf = confusion_matrix(y_test, mlp_pred).ravel() 
	dt_cf = confusion_matrix(y_test, dt_pred).ravel()

	M = {"tn": float(mlp_cf[0]), "fp": float(mlp_cf[1]), "fn": float(mlp_cf[2]), "tp": float(mlp_cf[3])}
	D = {"tn": float(dt_cf[0]), "fp": float(dt_cf[1]), "fn": float(dt_cf[2]), "tp": float(dt_cf[3])}
	
	# store error results in corresponding arrays
	mlp_errors.append((M["fp"] + M["fn"]) / float(sum(mlp_cf)))
	dt_errors.append((D["fp"] + D["fn"]) / float(sum(dt_cf)))

	# calculate precision, recall, and beta for F-measure
	try:
		mlp_precision = M["tp"] / (M["tp"] + M["fp"])
	except ValueError:
		mlp_precision = 0.0
	mlp_recall = M["tp"] / (M["tp"] + M["fn"])
	mlp_beta = M["fn"] / (M["fn"] + M["tp"])	
	try:
		dt_precision = D["tp"] / (D["tp"] + D["fp"])
	except ValueError:
		dt_precision = 0.0
	dt_recall = D["tp"] / (D["tp"] + D["fn"])
	dt_beta = D["fn"] / (D["fn"] + D["tp"])

	# calculate and store F-measures
	mlp_F = (pow(1 + mlp_beta,2) * mlp_precision * mlp_recall) / (pow(mlp_beta,2) * (mlp_precision + mlp_recall))
	mlp_Fs.append(mlp_F)
	dt_F = (pow(1 + dt_beta,2) * dt_precision * dt_recall) / (pow(dt_beta,2) * (dt_precision + dt_recall))
	dt_Fs.append(dt_F)

# print error results
print "MLP Error Scores:"
for e in mlp_errors:
	print "%0.3f" % e
print ""
print "Decision Tree Error Scores:"
for e in dt_errors:
	print "%0.3f" % e
	
# perform significance test with error as metric
T = 1.83

ED = [] # ED = Error Differences
for i in range(10):
	ED.append(mlp_errors[i] - dt_errors[i])
ED_mean = float(sum(ED)) / float(len(ED))
ED = map(lambda x: x - ED_mean, ED)
ED = map(lambda x: pow(x,2), ED)
S = float(pow(sum(ED) / 90., .5))
T_prime = ED_mean / S

print "For a 95% confidence interval, we use t = 1.83"
print "Using error as our metric, we get t' = %0.2f" % T_prime
print ""

# print F-measure results
print "MLP F-measures"
for F in mlp_Fs:
	print "%0.3f" % F
print ""
print "Decision Tree F-measures"
for F in dt_Fs:
	print "%0.3f" % F

# perform significance test with F-measure as metric
T = 1.83

FD = [] # F-measure differences
for i in range(10):
	FD.append(mlp_Fs[i] - dt_Fs[i])
FD_mean = float(sum(FD)) / float(len(FD))
FD = map(lambda x: x - FD_mean, FD)
FD = map(lambda x: pow(x,2), FD)
S = float(pow(sum(FD) / 90., .5))
T_prime = FD_mean / S

print "For a 95% confidence interval, we use t = 1.83"
print "Using F-measure as our metric, we get t' = %0.2f" % T_prime
