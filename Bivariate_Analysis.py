# Bivariate Analysis

# Load necessary packages

import pandas as pd
import numpy as np   # For mathematical calculations
import seaborn as sns   # For data visualization
import matplotlib.pyplot as plt   # For plotting graphs
import warnings   # To ignore any warnings
warnings.filterwarnings("ignore")

# Read data

train = pd.read_csv("/home/kostas/Documents/Loan Prediction set/train_ctrUa4K.csv")
test = pd.read_csv("/home/kostas/Documents/Loan Prediction set/test_lAUu6dG.csv")

# Copy train and test data to keep backups of the original datasets

train_original = train.copy()
test_original = test.copy()

# Categorical Independent Variable vs Target Variable (Loan Status)

# Gender vs Loan Status
Gender = pd.crosstab(train['Gender'],train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))


# Married vs Loan Status
Married = pd.crosstab(train['Married'],train['Loan_Status'])
Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked= True, figsize=(4,4))

# Dependents vs Loan Status
Dependents = pd.crosstab(train['Dependents'],train['Loan_Status'])
Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked= True, figsize=(4,4))

# Education vs Loan Status
Education = pd.crosstab(train['Education'],train['Loan_Status'])
Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked= True, figsize=(4,4))

# Self Employed vs Loan Status
Self_employed = pd.crosstab(train['Self_Employed'],train['Loan_Status'])
Self_employed.div(Self_employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked= True, figsize=(4,4))

# Credit History vs Loan Status
Credit_History = pd.crosstab(train['Credit_History'],train['Loan_Status'])
Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked= True, figsize=(4,4))

# Property Area vs Loan Status
Property_Area = pd.crosstab(train['Property_Area'],train['Loan_Status'])
Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked= True, figsize=(4,4))

# Nmerical Independent Variable vs Target Variable (Loan Status)

# Applicant Income vs Loan Status
train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()

# Create bins/groups for Applicant's Income
bins = [0,2500,4000,6000,81000]
group = ['Low','Average','High','Very High']

train['Income_bin'] = pd.cut(train['ApplicantIncome'],bins,labels=group)

# Grouped Applicant Income vs Loan Status
Income_bin = pd.crosstab(train['Income_bin'],train['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True)
plt.xlabel('ApplicantIncome')
plt.ylabel('Percentage')

# Co Applicant Income vs Loan Status
# Create bins/groups for Co Applicant's Income
bins = [0,1000,3000,42000]
group = ['Low','Average','High']

train['Coapplicant_Income_bin'] = pd.cut(train['CoapplicantIncome'],bins,labels=group)

# Grouped Co Applicant Income vs Loan Status
Coapplicant_Income_bin = pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'])
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True)
plt.xlabel("CoapplicantIncome")
plt.ylabel('Percentage')

# Combine Applicant and Co Applicant Income
train['Total_Income'] = train['ApplicantIncome'] + train['CoapplicantIncome']

# Create bins/groups for Total Income
bins = [0,2500,4000,6000,81000]
group = ['Low','Average','High','Very High']

train['Total_Income_bin'] = pd.cut(train['Total_Income'],bins,labels=group)

# Grouped Total Income vs Loan Status
Total_Income_bin = pd.crosstab(train['Total_Income_bin'],train['Loan_Status'])
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True)
plt.xlabel('Total_Income')
plt.ylabel('Percentage')

# Loan Amount vs Loan Status
# Create bins/groups for Loan Amount
bins = [0,100,200,700]
group = ['Low','Average','High']

train['LoanAmount_bin'] = pd.cut(train['LoanAmount'],bins,labels=group)

# Grouped Loan Amount vs Loan Status
Loan_Amount_bin = pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'])
Loan_Amount_bin.div(Loan_Amount_bin.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True)
plt.xlabel('LoanAmount')
plt.ylabel('Percentage')

# Visualize the dependency between all variables.
# Non numerical variables will be changed to numerical.

# Drop the bins which were created for the exploration part from the train data set
train = train.drop(['Income_bin', 'Coapplicant_Income_bin',
                    'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis = 1)

# Replace "3+" in Dependents variable with 3
# Replace "N" with 0 and "Y" with 1 in Loan_Status variable
train['Dependents'].replace('3+', 3, inplace=True)
test['Dependents'].replace('3+', 3, inplace=True)

train['Loan_Status'].replace('N', 0, inplace=True)
train['Loan_Status'].replace('Y', 1, inplace=True)

# Create heat map to visualize correlation
matrix = train.corr()
f, ax = plt.subplots(figsize = (9,6))
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu")

# Show all figures
plt.show()