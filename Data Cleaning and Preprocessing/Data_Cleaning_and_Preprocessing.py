### Note :- The data set is almost clean and ready for data analysis, I have written all these code lines for showing how we perform data cleaning, if the data set is not proper.
## Importing libraries 
import pandas as pd
import numpy as np

file_path = "Customer Retention Analysis for an E-commerce Business.xlsx"     ## Path to the excel file 
df = pd.read_excel(file_path, sheet_name="Customer_Retention_Data")           ## Load the Excel file

print(df.head())   # Check first 5 rows

print(df.info())    # Summary of data types and non-null counts

print(df.describe())  # Descriptive statistics for numerical columns

print(df[['Country', 'State', 'Gender', 'Churned']].nunique())   # Check unique values in categorical columns

print("Missing Values:\n", df.isnull().sum())    # Check for missing values

#####  Handle missing values:

df = df.dropna(subset=['Customer_ID', 'Churned'])     # - Drop rows with missing critical fields (e.g., Customer_ID, Churned)

df['Age'] = df['Age'].fillna(df['Age'].median())          # Impute numerical columns (e.g., Age) with median to avoid outlier influence

df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])    # Impute categorical columns (e.g., Gender) with mode

print("Duplicate Customer_IDs:", df.duplicated(subset=['Customer_ID']).sum())    # Check for duplicate Customer_IDs

df = df.drop_duplicates(subset=['Customer_ID'], keep='first')    # Drop duplicates (keep first occurrence)

####  Standardize categorical columns

df['Gender'] = df['Gender'].str.capitalize()           # Ensure 'Male'/'Female'
df['Country'] = df['Country'].str.strip().str.title()  # Fix country name casing
df['State'] = df['State'].str.strip().str.title()      # Fix state name casing

df['Churned'] = df['Churned'].replace({'yes': 'Yes', 'no': 'No'}).str.strip().str.title()   # Ensure 'Churned' is binary (Yes/No)

df['Age'] = df['Age'].astype(int)
df['Purchase_Frequency'] = df['Purchase_Frequency'].astype(float)
df['Avg_Purchase_Value'] = df['Avg_Purchase_Value'].astype(float)
df['Last_Purchase_Days_Ago'] = df['Last_Purchase_Days_Ago'].astype(int)

# Ensure no negative values in numerical fields
df = df[df['Age'] > 0]
df = df[df['Last_Purchase_Days_Ago'] >= 0]

# Check for outliers 
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Age'] < (Q1 - 1.5 * IQR)) | (df['Age'] > (Q3 + 1.5 * IQR)))]

df.to_excel("Cleaned_Customer_Retention_Data.xlsx", index=False)   # Save cleaned data to a new Excel file



### OUTPUT for all lines 

# Customer_ID        Name       Country   State  Gender  Age  Purchase_Frequency  Avg_Purchase_Value  Last_Purchase_Days_Ago Churned
# 0   CUST00001  Customer_1       Nigeria    Kano  Female   23                  20               58.87                     201      No
# 1   CUST00002  Customer_2       Nigeria    Kano    Male   59                  44              324.55                     313     Yes
# 2   CUST00003  Customer_3  South Africa  Durban    Male   41                  34               45.58                     292     Yes
# 3   CUST00004  Customer_4         Kenya  Kisumu    Male   34                  40              143.90                     304      No
# 4   CUST00005  Customer_5         Kenya  Kisumu    Male   23                  26              414.09                     274     Yes
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1000 entries, 0 to 999
# Data columns (total 10 columns):
#  #   Column                  Non-Null Count  Dtype
# ---  ------                  --------------  -----
#  0   Customer_ID             1000 non-null   object
#  1   Name                    1000 non-null   object
#  2   Country                 1000 non-null   object
#  3   State                   1000 non-null   object
#  4   Gender                  1000 non-null   object
#  5   Age                     1000 non-null   int64
#  6   Purchase_Frequency      1000 non-null   int64
#  7   Avg_Purchase_Value      1000 non-null   float64
#  8   Last_Purchase_Days_Ago  1000 non-null   int64
#  9   Churned                 1000 non-null   object
# dtypes: float64(1), int64(3), object(6)
# memory usage: 78.3+ KB
# None
#                Age  Purchase_Frequency  Avg_Purchase_Value  Last_Purchase_Days_Ago
# count  1000.000000         1000.000000         1000.000000             1000.000000
# mean     40.822000           26.567000          256.295600              189.290000
# std      14.051896           14.688201          138.791381              108.147722
# min      18.000000            1.000000            5.100000                0.000000
# 25%      28.000000           13.000000          134.257500               91.000000
# 50%      40.500000           27.000000          263.260000              195.500000
# 75%      53.250000           40.000000          373.570000              285.000000
# max      65.000000           50.000000          499.980000              365.000000
# Country     5
# State      16
# Gender      2
# Churned     2
# dtype: int64
# Missing Values:
#  Customer_ID               0
# Name                      0
# Country                   0
# State                     0
# Gender                    0
# Age                       0
# Purchase_Frequency        0
# Avg_Purchase_Value        0
# Last_Purchase_Days_Ago    0
# Churned                   0
# dtype: int64
# Duplicate Customer_IDs: 0