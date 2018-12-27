'''
Accuracy of classifiers for Wine Dataset (UCI)
-- Sandeep Kundala
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB

from sklearn import metrics


wine_data = pd.read_csv('C:/Users/Sandi/Downloads/Wine.csv',
                           sep= ',', header= None)
wine_data.columns = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

#DTC
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 0,
                               min_samples_split =3)
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 0,
                               min_samples_split =3)

#KNN
knn = KNeighborsClassifier(n_neighbors=5)

#NAIVE BAYES
gnb = GaussianNB()
mnb = MultinomialNB()
cnb = ComplementNB()

#------- BEFORE CLEANING --------#
X_bc = wine_data.values[:, 1:13]
Y_bc = wine_data.values[:,0]
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split( X_bc, Y_bc, test_size = 0.2, random_state=0)

#-------------DTC-------------------------------------------------------------

clf_gini.fit(X_train_bc, y_train_bc)
clf_entropy.fit(X_train_bc, y_train_bc)
y_pred_bc_DTC_gini = clf_gini.predict(X_test_bc)
y_pred_bc_DTC_entropy = clf_entropy.predict(X_test_bc)

#-------------KNN-------------------------------------------------------------
knn.fit(X_train_bc, y_train_bc)
y_pred_bc_knn = knn.predict(X_test_bc)

#------------NAIVE BAYES------------------------------------------------------
y_pred_bc_gnb = gnb.fit(X_train_bc, y_train_bc).predict(X_test_bc)
y_pred_bc_mnb = mnb.fit(X_train_bc, y_train_bc).predict(X_test_bc)
y_pred_bc_cnb = cnb.fit(X_train_bc, y_train_bc).predict(X_test_bc)

#----------AFTER CLEANING-----------------------------------------------------

to_drop = ['Malic acid', 'Ash',  'Magnesium', 'Total phenols', 'Nonflavanoid phenols', 'Proanthocyanins']

wine_data.drop(to_drop, inplace=True, axis=1)
X_ac = wine_data.values[:, 1:6]
Y_ac = wine_data.values[:,0]
X_train_ac, X_test_ac, y_train_ac, y_test_ac = train_test_split( X_ac, Y_ac, test_size = 0.2, random_state=0)

#------------------DTC--------------------------------------------------------
clf_gini.fit(X_train_ac, y_train_ac)
clf_entropy.fit(X_train_ac, y_train_ac)
y_pred_ac_DTC_gini = clf_gini.predict(X_test_ac)
y_pred_ac_DTC_entropy = clf_entropy.predict(X_test_ac)

#-------------KNN-------------------------------------------------------------
knn.fit(X_train_ac, y_train_ac)
y_pred_ac_knn = knn.predict(X_test_ac)

#------------NAIVE BAYES------------------------------------------------------
#
y_pred_ac_gnb = gnb.fit(X_train_ac, y_train_ac).predict(X_test_ac)
y_pred_ac_mnb = mnb.fit(X_train_ac, y_train_ac).predict(X_test_ac)
y_pred_ac_cnb = cnb.fit(X_train_ac, y_train_ac).predict(X_test_ac)

print("BEFORE CLEANING:")
print("Accuracy of DTC (gini): ", metrics.accuracy_score(y_test_bc,y_pred_bc_DTC_gini)*100)
print("Accuracy of DTC (entropy):", metrics.accuracy_score(y_test_bc,y_pred_bc_DTC_entropy)*100)
print("Accuracy of KNN:",metrics.accuracy_score(y_test_bc, y_pred_bc_knn)*100)
print("Accuracy of GNB:",metrics.accuracy_score(y_test_bc, y_pred_bc_gnb)*100)
print("Accuracy of MNB:",metrics.accuracy_score(y_test_bc, y_pred_bc_mnb)*100)
print("Accuracy of CNB:",metrics.accuracy_score(y_test_bc, y_pred_bc_cnb)*100)
print("AFTER CLEANING:")
print("Accuracy of DTC (gini):", metrics.accuracy_score(y_test_ac,y_pred_ac_DTC_gini)*100)
print("Accuracy of DTC (entropy):", metrics.accuracy_score(y_test_ac,y_pred_ac_DTC_entropy)*100)
print("Accuracy of KNN:",metrics.accuracy_score(y_test_ac, y_pred_ac_knn)*100)
print("Accuracy of GNB:",metrics.accuracy_score(y_test_ac, y_pred_ac_gnb)*100)
print("Accuracy of MNB:",metrics.accuracy_score(y_test_ac, y_pred_ac_mnb)*100)
print("Accuracy of CNB:",metrics.accuracy_score(y_test_ac, y_pred_ac_cnb)*100)
