# Exploratory Data Analysis (EDA)

# Load necessary packages

import pandas as pd
import warnings   # To ignore any warnings
warnings.filterwarnings("ignore")

# Read data

train = pd.read_csv("/home/kostas/Documents/Loan Prediction set/train_ctrUa4K.csv")
test = pd.read_csv("/home/kostas/Documents/Loan Prediction set/test_lAUu6dG.csv")

# Copy train and test data to keep backups of the original data sets

train_original = train.copy()
test_original = test.copy()

# Check the features present at our data sets

train_columns = train.columns
test_columns = test.columns

print(train_columns)
print(test_columns)

# The Target Variable is Loan_Status.
# Use the train dataset to predict the load status for the test dataset.

# Print data types for each variable
# * object: Object format means variables are categorical.
# Categorical variables in our dataset are:
    # Loan_ID,
    # Gender,
    # Married,
    # Dependents,
    # Education,
    # Self_Employed,
    # Property_Area,
    # Loan_Status
# * int64: It represents the integer variables.
    # ApplicantIncome
# * float64: It represents the variable which have some decimal values involved.
# They are also numerical variables.
# Numerical variables in our dataset are:
    # CoapplicantIncome,
    # LoanAmount,
    # Loan_Amount_Term,
    # Credit_History

train_dtypes = train.dtypes
test_dtypes = test.dtypes

print(train_dtypes)
print(test_dtypes)

# Shape of the data sets

train_shape = train.shape
test_shape = test.shape

print(train_shape)
print(test_shape)