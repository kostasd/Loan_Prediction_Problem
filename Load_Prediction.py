# This script was created for the Analytics Vidhya Loan Prediction Practice Problem.
# It will automate the loan eligibility process based on customer details.
# Details used:
# * Gender
# * Marital Status
# * Education
# * Number of Dependents
# * Employment Status
# * Income (Applicant and Co-Applicant)
# * Loan Amount
# * Loan Amount Term
# * Credit History
# * Property Area

# Load necessary packages

import pandas as pd
import numpy as np   # For mathematical calculations
import matplotlib.pyplot as plt   # For plotting graphs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import warnings   # To ignore any warnings
warnings.filterwarnings("ignore")

# Read data

train = pd.read_csv("/home/kostas/Documents/Loan Prediction set/train_ctrUa4K.csv")
test = pd.read_csv("/home/kostas/Documents/Loan Prediction set/test_lAUu6dG.csv")

# Copy train and test data to keep backups of the original datasets

train_original = train.copy()
test_original = test.copy()

# Missing Value treatment

table_missing_val = train.isnull().sum()

print(table_missing_val)

# Fill missing values using the Mode value for categorical variables
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)

# Fill missing values using media for Loan Amount as there are many outliers
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

# Check if all missing values were filled
table_missing_val = train.isnull().sum()

print(table_missing_val)

# Fill the missing values in test data set using the same approach
test['Gender'].fillna(test['Gender'].mode()[0], inplace=True)
test['Married'].fillna(test['Married'].mode()[0], inplace=True)
test['Dependents'].fillna(test['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(test['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(test['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(test['LoanAmount'].median(), inplace=True)

# Outlier treatment

# Remove outliers from Loan Amount variable
# Use log transformation to reduce larger values and remove skewness
train['LoanAmount_log'] = np.log(train['LoanAmount'])
test['LoanAmount_log'] = np.log(test['LoanAmount'])

train['LoanAmount_log'].hist(bins=20)
plt.show(block=False)
plt.pause(3)
plt.close()

# Prediction Model
# Linear Logistic Regression

# Drop Loan_ID variable as it is not needed
train = train.drop('Loan_ID', axis=1)
test = test.drop('Loan_ID', axis=1)

# Save our target variable to a different data set
X = train.drop('Loan_Status', 1)
y = train.Loan_Status

# Create dummy variables to change categorical variables to numerical
X = pd.get_dummies(X)
train_dummies = pd.get_dummies(train)
test_dummies = pd.get_dummies(test)

# Split train data set into train and validation
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size=0.2)

# Create model
model = LogisticRegression()
model.fit(x_train, y_train)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                   penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
                   verbose=0, warm_start=False)

# Predict the Loan Status for validation set
pred_cv = model.predict(x_cv)

# Evaluation of Model
accuracy_logistic = accuracy_score(y_cv, pred_cv)

print(accuracy_logistic)

# Predict the Loan Status for test data set
pred_test = model.predict(test_dummies)

# Submit results for Analytics Vidhya
submission = pd.read_csv("/home/kostas/Documents/Loan Prediction set/sample_submission_49d68Cx.csv")

submission['Loan_Status'] = pred_test
submission['Loan_ID'] = test_original['Loan_ID']

# Convert 1 and 0 to "Y" and "N" for Loan Status variable
submission['Loan_Status'].replace(0, 'N', inplace=True)
submission['Loan_Status'].replace(1, 'Y', inplace=True)

# Convert submission to .csv
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('output/logistic.csv', index=False)

# Logistic Regression using stratified k-fold cross validation
i = 1
kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

for train_index, test_index in kf.split(X,y):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr, xvl = X.loc[train_index], X.loc[test_index]
    ytr, yvl = y[train_index], y[test_index]
    model = LogisticRegression(random_state=1)
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    print('accuracy_score', score)
    i += 1
    pred_test = model.predict(test_dummies)
    pred = model.predict_proba(xvl)[:,1]

# Submit results for Analytics Vidhya
submission['Loan_Status'] = pred_test
submission['Loan_ID'] = test_original['Loan_ID']

# Convert 1 and 0 to "Y" and "N" for Loan Status variable
submission['Loan_Status'].replace(0, 'N', inplace=True)
submission['Loan_Status'].replace(1, 'Y', inplace=True)

# Convert submission to .csv
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('output/Logistic.csv', index=False)

# Add three new variables in the dataset to improve accuracy

# Total Income -
# how much money Applicant and Co-Applicant make
train['Total_Income'] = train['ApplicantIncome'] + train['CoapplicantIncome']
test['Total_Income'] = test['ApplicantIncome'] + test['CoapplicantIncome']

# Normalize Total Income
train['Total_Income_log'] = np.log(train['Total_Income'])
test['Total_Income_log'] = np.log(test['Total_Income'])

# EMI (Estimated Monthly Installment) -
# how much money the person has to pay for the loan
train['EMI'] = train['LoanAmount'] / train['Loan_Amount_Term']
test['EMI'] = test['LoanAmount'] / test['Loan_Amount_Term']

# Balance Income -
# how much money left after paying monthly payment
train['Balance_Income'] = train['Total_Income'] - (train['EMI']*1000)
test['Balance_Income'] = test['Total_Income'] - (test['EMI']*1000)

# Drop all variables used to create the new 3 variables
train_new = train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)
test_new = test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)

# Logistic Regression after the addition of the 3 variables

# Save our target variable to a different data set
X_new = train_new.drop('Loan_Status', 1)
y_new = train_new.Loan_Status

# Create dummy variables to change categorical variables to numerical
X_new = pd.get_dummies(X_new)
train_new_dummies = pd.get_dummies(train_new)
test_new_dummies = pd.get_dummies(test_new)

# Split train data set into train and validation
x_train, x_cv, y_train, y_cv = train_test_split(X_new,y_new, test_size=0.2)

# Create model
model = LogisticRegression()
model.fit(x_train, y_train)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                   penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
                   verbose=0, warm_start=False)

# Predict the Loan Status for validation set
pred_cv = model.predict(x_cv)

# Evaluation of Model
accuracy_logistic_new = accuracy_score(y_cv, pred_cv)

print(accuracy_logistic_new)

# Predict the Loan Status for test data set
pred_test = model.predict(test_new_dummies)

# Submit results for Analytics Vidhya
submission = pd.read_csv("/home/kostas/Documents/Loan Prediction set/sample_submission_49d68Cx.csv")

submission['Loan_Status'] = pred_test
submission['Loan_ID'] = test_original['Loan_ID']

# Convert 1 and 0 to "Y" and "N" for Loan Status variable
submission['Loan_Status'].replace(0, 'N', inplace=True)
submission['Loan_Status'].replace(1, 'Y', inplace=True)

# Convert submission to .csv
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('output/logistic_new.csv' ,index=False)

# Logistic Regression using stratified k-fold cross validation
i = 1
kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

for train_index, test_index in kf.split(X_new,y_new):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr, xvl = X_new.loc[train_index], X_new.loc[test_index]
    ytr, yvl = y_new[train_index], y_new[test_index]
    model = LogisticRegression(random_state=1)
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    print('accuracy_score', score)
    i += 1
    pred_test = model.predict(test_new_dummies)
    pred = model.predict_proba(xvl)[:,1]

# Submit results for Analytics Vidhya
submission['Loan_Status'] = pred_test
submission['Loan_ID'] = test_original['Loan_ID']

# Convert 1 and 0 to "Y" and "N" for Loan Status variable
submission['Loan_Status'].replace(0, 'N', inplace=True)
submission['Loan_Status'].replace(1, 'Y', inplace=True)

# Convert submission to .csv
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('output/Logistic_new.csv', index=False)

plt.show(block=False)
plt.pause(3)
plt.close()

# XGBoost algorithm

i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)

for train_index,test_index in kf.split(X_new,y_new):
    print('\n{} of kfold {}'.format(i, kf.n_splits))
    xtr, xvl = X_new.loc[train_index], X_new.loc[test_index]
    ytr, yvl = y_new[train_index], y_new[test_index]
    model = XGBClassifier(n_estimators=50, max_depth=4)
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    print('accuracy_score', score)
    i += 1
    pred_test = model.predict(test_new_dummies)
    pred = model.predict_proba(test_new_dummies)[:, 1]

# Submit results for Analytics Vidhya
submission['Loan_Status'] = pred_test
submission['Loan_ID'] = test_original['Loan_ID']

# Convert 1 and 0 to "Y" and "N" for Loan Status variable
submission['Loan_Status'].replace(0, 'N', inplace=True)
submission['Loan_Status'].replace(1, 'Y', inplace=True)

# Convert submission to .csv
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('output/XGBoost.csv', index=False)
