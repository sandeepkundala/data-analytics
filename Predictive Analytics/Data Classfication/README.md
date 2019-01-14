This section has been referred from Introduction to Data Mining, Tan et. al., some parts have been copied.

## Data Classification [1]
Examples of classification -
1. Predicting tumor cells as benign or malignant
2. Classifying credit card transactions as legitimate or fraudulent
3. Categorizing news stories as finance, weather, entertainment, sports, etc.

To understand data classification, we must understand how the data is represented. A dataset is a combination of rows and columns. Where rows are the data points and columns are the attributes or features of the data. 
### What is classification?
Classification or any other data analytics task uses two methods -
1. Induction - This is a process of learning a model using a training dataset.
2. Deduction - This is a process of using the learned model on test data set. In terms of classification, this involves identifying the class or label associated with all data points in a test set.

Formally,
Given a collection of records i.e. training set. Each record od data point consists of multiple attributes, one such attribute is the class. The objective is to develop a classification model for the class attribute/label as a function of the rest of the attributes.

There can be two goals associated with a classification task.
1. Describe data or descriptive modelling.
2. Assign previously unseen records a class as accurately as possible i.e. predictive modelling.

### Classification metrics? 
1. Accuracy is the most widely used performance evaluation metric. It is defined as (TP + TN)/(TP + TN + FP + FN). Classification error is = 1 - accuracy.

### Classification methods?

1. #### Decision Tree based Methods
2. #### Rule-based Methods
3. #### Memory based reasoning
4. #### Neural Networks
5. #### Naïve Bayes
6. #### Support Vector Machines

### Issues of Classification
1. #### Underfitting and Overfitting

Overfitting -  Model fits the training data well, but not the test dataset. This is mainly due to noise in the data or lack of representative records for some classes. If there is noise in the data the decision boundary may split because of that noise data. However, if there is lack of representative records for some classes, those data points will be misclassified on the test data. Overfitting can't be estimated on the basis of training error. We may need to prune decision trees if we are using decision trees for classification.

Underfitting - Model is too simple, with high errors on the training as well as the test dataset.

2. #### Missing Values
3. #### Cost of classification

[1] Introduction to Data Mining by Tan, Steinbach, Kumar



