import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

data = pd.read_csv("ism.csv")
data = data.as_matrix()

X = data[:,:6]
y = data[:,6]

#X_train, X_test, y_train, y_test = train_test_split(data[:,:6],data[:,6],test_size=0.2)

mlp = MLPClassifier(solver = 'sgd', learning_rate_init=.01)
mlp.fit(X, y)

dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(X, y)

skf = StratifiedKFold(n_splits=10)


mlp_scores = cross_val_score(mlp, X, y, cv = 10)
dt_scores = cross_val_score(dt, X, y, cv = 10)

print "MLP Scores:"
for s in mlp_scores:
	print s
print ""
print ("Accuracy: %0.2f +/- %0.2f" % (mlp_scores.mean(), mlp_scores.std() * 2))
print ""

print "Decision Tree Scores:"
for s in dt_scores:
	print s
print ""
print ("Accuracy: %0.2f +/- %0.2f" % (dt_scores.mean(), mlp_scores.std() * 2))
print ""
