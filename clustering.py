import pandas as pd
import numpy as np
#import ML support libraries

from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def loadData(datafile):
        with open(datafile,'r', encoding="Latin-1") as csvfile:
            data = pd.read_csv(csvfile)

        #Inspect the data
        print(data.columns.values)
        return data

def runKNN(dataset, prediction, ignore, k):
    # Set up our datasets
    X = dataset.drop(columns=[prediction, ignore])
    Y = dataset[prediction].values
    # Split the data into a training and testing set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1, stratify=Y)

    #Run k-NN algorithm
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)

    # Test the model
    score = knn.score(X_test, Y_test)
    Y_pred = knn.predict(X_test)
    print("Predicts " + prediction + " with " + str(score) + " accuracy")
    print("Chance is: " + str(1.0/len(dataset.groupby(prediction))))
    print("F1 score: " + str(f1_score(Y_test, Y_pred, average='macro')))

    return knn

def classifyPlayer(targetRow, data, model, prediction, ignore):
    X = targetRow.drop(columns = [prediction, ignore])
    #Dtermine five closest neighbors
    neighbors = model.kneighbors(X, n_neighbors=5, return_distance=False)

    #Print out the neighbors data
    for neighbor in neighbors[0]:
        print(data.iloc[neighbor])

def kNNCrossfold(dataset, prediction, ignore, k):
    fold = 0
    accuracies = []
    kf = KFold(n_splits=k)

    X = dataset.drop(columns=[prediction, ignore])
    Y = dataset[prediction].values

    for train,test in kf.split(X):
        fold += 1
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X[train[0]:train[-1]], Y[train[0]:train[-1]])

        pred = knn.predict(X[test[0]:test[-1]])
        accuracy = accuracy_score(pred, Y[test[0]:test[-1]])
        accuracies.append(accuracy)
        print("Fold " + str(fold) + ":" + str(accuracy))

    return np.mean(accuracies)


def determineK(dataset, prediction, ignore, k_vals):
    best_k = 0
    best_accuracy = 0

    for k in k_vals:
        current_k = kNNCrossfold(dataset, prediction, ignore, k)
        if current_k > best_accuracy:
            best_k = k
            best_accuracy = current_k

    print("Best k, accuracy = " + str(best_k) + ", " + str(best_accuracy))



nbaData = loadData("nba_2013_clean.csv")
knnModel= runKNN(nbaData, "pos", "player", 5)
classifyPlayer(nbaData.loc[nbaData['player'] == "LeBron James"], nbaData, knnModel, 'pos', 'player')
for k in [5,7,10]:
    print("Folds: " + str(k))
    kNNCrossfold(nbaData,"pos", "player", k)

determineK(nbaData,"pos", "player", [5,7,10])
#Problem 2
# The classifier has about a 50% chance of predicting a player's position correctly. This is an okay-performing model. It shouldn't be used to make crucial decisions.abs
