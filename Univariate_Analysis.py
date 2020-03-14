# Univariate analysis

# Load necessary packages

import pandas as pd
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

# For Categorical features use frequency tables and bar plots
# For Numerical features use probability density plots or boxplots

# Categorical variables

# Loan_status

Loan_status_counts = train['Loan_Status'].value_counts()

print(Loan_status_counts) # There are no missing values for the loan status

# Normalize can be set to True to print proportions instead of number

Loan_status_prop = train['Loan_Status'].value_counts(normalize=True)

print(Loan_status_prop) # 68.73% has approved loan / 31.27% has rejected loan

# Bar plot for Loan Status train dataset

Loan_status_prop.plot.bar()
plt.show()

# Gender, Married, Self Employed, Credit History

plt.figure(1)

plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title='Gender')

plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(title='Married')

plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(title='Self_Employed')

plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(title='Credit_History')

plt.show()

# Based on the above data
    # 80% applicants in the dataset are male
    # Around 65% of the applicants in the dataset are married
    # Around 15% applicants in the dataset are self employed
    # Around 85% applicants have repaid their debts

# Ordinal (Categorical) variables

plt.figure(2)

plt.subplot(131)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6), title='Dependents')

plt.subplot(132)
train['Education'].value_counts(normalize=True).plot.bar(title='Education')

plt.subplot(133)
train['Property_Area'].value_counts(normalize=True).plot.bar(title='Property_Area')

plt.show()

# Based on the above data
    # Most applicants do not have any dependents
    # Around 80% of the applicants are Graduate
    # Most of the applicants are from Semiurban area

# Numerical variables

plt.figure(3)

plt.subplot(121)
sns.distplot(train['ApplicantIncome']);

plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize=(16,5))

train.boxplot(column='ApplicantIncome', by='Education')
plt.suptitle("")

plt.show()

# The applicant income is not normally distributed
# There are a lot of outliers

plt.figure(4)

plt.subplot(121)
sns.distplot(train['CoapplicantIncome']);

plt.subplot(122)
train['CoapplicantIncome'].plot.box(figsize=(16,5))

plt.show()

# The Coapplicant income is not normally distributed
# There are a lot of outliers

plt.figure(5)

plt.subplot(121)
df=train.dropna() # Not necessary
sns.distplot(train['LoanAmount']);

plt.subplot(122)
train['LoanAmount'].plot.box(figsize=(16,5))

plt.show()

# THe Loan Amount has fairly normal distribution
# There are a lot of outliers