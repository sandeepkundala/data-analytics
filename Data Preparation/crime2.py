'''
Accuracy of classifiers for Crime Dataset
Crime Classification dataset
San Fransisco Police Department Crime Classification

-- Sandeep Kundala
'''

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer

crime_data = pd.read_csv('C:/Users/Sandi/Downloads/Crime1.csv',
                           sep= ',', header= 0)

columnsTitles=['Category', 'Dates', 'Descript', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y']
crime_data=crime_data.reindex(columns=columnsTitles)
crime_data= crime_data[:3000]
# new data frame with split value columns 
new = crime_data["Dates"].str.split(" ", n = 1, expand = True) 
  
# making seperate date column from Dates Column 
crime_data["Date"]= new[0] 
  
# making seperate time column from Dates Column 
crime_data["Time"]= new[1] 
  
# Dropping old Dates columns 
crime_data.drop(columns =["Dates"], inplace = True) 

#To obtain correlation between different columns, we need to map the strings to a constant.

crime_data['category_id'] = crime_data['Category'].factorize()[0]
crime_data['Date_id'] = crime_data['Date'].factorize()[0]
crime_data['Time_id'] = crime_data['Time'].factorize()[0]
crime_data['DayOfWeek_id'] = crime_data['DayOfWeek'].factorize()[0]
crime_data['PdDistrict_id'] = crime_data['PdDistrict'].factorize()[0]
crime_data['Resolution_id'] = crime_data['Resolution'].factorize()[0]
crime_data['Address_id'] = crime_data['Address'].factorize()[0]
columnsTitles=['Category', 'category_id', 'Date', 'Date_id','Time','Time_id', 'Descript', 'DayOfWeek', 'DayOfWeek_id','PdDistrict','PdDistrict_id', 'Resolution', 'Resolution_id','Address','Address_id', 'X', 'Y']
crime_data=crime_data.reindex(columns=columnsTitles)

correlations = crime_data[crime_data.columns].corr(method='pearson')
sns.heatmap(correlations, cmap="YlGnBu", annot = True)

# we see that there's not much correlation between the other columns and the category so we would consider the other approach i.e., use "Descript" column from
# Crime1.csv and search for keywords specific to the crime

category_id_df = crime_data[['Category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Category']].values)

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(crime_data['Descript']).toarray()
labels = crime_data['category_id']
features.shape


N = 2
for Category, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(Category))
  print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))
  
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, crime_data.index, test_size=0.2, random_state=0)

#Linear SVC
lsvc = LinearSVC()
y_pred_LSVC = lsvc.fit(X_train, y_train).predict(X_test)
print("Accuracy of SVC: ", metrics.accuracy_score(y_test,y_pred_LSVC)*100)

#DTC
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 0,
                               min_samples_split =3)
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 0,
                               min_samples_split =3)
y_pred_DTC_gini = clf_gini.fit(X_train, y_train).predict(X_test)
y_pred_DTC_entropy = clf_entropy.fit(X_train, y_train).predict(X_test)
print("Accuracy of DTC (gini): ", metrics.accuracy_score(y_test,y_pred_DTC_gini)*100)
print("Accuracy of DTC (entropy):", metrics.accuracy_score(y_test,y_pred_DTC_entropy)*100)

#KNN
knn = KNeighborsClassifier(n_neighbors=25)
y_pred_knn = knn.fit(X_train, y_train).predict(X_test)
print("Accuracy of KNN:",metrics.accuracy_score(y_test, y_pred_knn)*100)

#NAIVE BAYES
gnb = GaussianNB()
mnb = MultinomialNB()
cnb = ComplementNB()

y_pred_gnb = gnb.fit(X_train, y_train).predict(X_test)
y_pred_mnb = mnb.fit(X_train, y_train).predict(X_test)
y_pred_cnb = cnb.fit(X_train, y_train).predict(X_test)

print("Accuracy of GNB:",metrics.accuracy_score(y_test, y_pred_gnb)*100)
print("Accuracy of MNB:",metrics.accuracy_score(y_test, y_pred_mnb)*100)
print("Accuracy of CNB:",metrics.accuracy_score(y_test, y_pred_cnb)*100)

#Logistic Regression
lr = LogisticRegression(random_state=0)
y_pred_lc = lr.fit(X_train, y_train).predict(X_test)
print("Accuracy of Logistic Regression:",metrics.accuracy_score(y_test, y_pred_lc)*100)

#Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)
y_pred_rfc = rfc.fit(X_train, y_train).predict(X_test)
print("Accuracy of Random Forest Classifier:",metrics.accuracy_score(y_test, y_pred_rfc)*100)
