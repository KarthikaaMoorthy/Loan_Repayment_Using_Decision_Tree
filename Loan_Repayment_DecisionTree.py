

#Loan Repayment Prediction

#import necessary packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

balance_data = pd.read_csv(r"C:\Users\Admin\Desktop\Loan_Repayment_Dataset.csv")

#separate dependent and independent variables
X = balance_data.values[:, 1:5]
Y = balance_data.values[:,0]

#split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.3, random_state= 100)

#train the model using decisiontree classifier
clf_entropy = DecisionTreeClassifier(criterion= "entropy", random_state=100, max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, Y_train)

#predict the test data
y_pred = clf_entropy.predict(X_test)

#calculate accuracy of the model
print("Accuracy is" , accuracy_score(Y_test, y_pred)*100)
