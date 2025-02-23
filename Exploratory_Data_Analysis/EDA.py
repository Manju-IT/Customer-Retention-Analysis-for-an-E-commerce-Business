## Importing required librabries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory for saving visualizations
os.makedirs("visualizations", exist_ok=True)

# Load cleaned dataset
df = pd.read_excel("Cleaned_Customer_Retention_Data.xlsx")

# Set style
sns.set_theme(style="whitegrid", palette="pastel")


# 1. Age Distribution (Histogram)

plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=20, kde=True, color='skyblue')
plt.title("Age Distribution of Customers", fontsize=14, pad=20)
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("visualizations/age_distribution.png", bbox_inches='tight')
plt.close()


# 2. Gender Split (Bar Chart)

gender_counts = df['Gender'].value_counts()
plt.figure(figsize=(8, 6))
gender_counts.plot(kind='bar', color=['lightcoral', 'steelblue'])
plt.title("Customer Gender Distribution", fontsize=14, pad=20)
plt.xlabel("Gender")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.savefig("visualizations/gender_split.png", bbox_inches='tight')
plt.close()


# 3. Country-Level Churn (Stacked Bar)

country_churn = df.groupby('Country')['Churned'].value_counts(normalize=True).unstack()
plt.figure(figsize=(12, 6))
country_churn.plot(kind='bar', stacked=True, color=['#4c72b0', '#c44e52'])
plt.title("Churn Rate by Country", fontsize=14, pad=20)
plt.xlabel("Country")
plt.ylabel("Proportion")
plt.legend(title='Churned', loc='upper right')
plt.savefig("visualizations/country_churn.png", bbox_inches='tight')
plt.close()


# 4. Purchase Frequency vs Churn (Box Plot)

plt.figure(figsize=(10, 6))
sns.boxplot(
    x='Churned', 
    y='Purchase_Frequency', 
    data=df, 
    hue='Churned',  # Added hue parameter
    palette=['#4c72b0', '#c44e52'],
    legend=False  # Disables redundant legend
)
plt.title("Purchase Frequency vs Churn Status", fontsize=14, pad=20)
plt.xlabel("Churned")
plt.ylabel("Purchase Frequency (times/year)")
plt.savefig("visualizations/purchase_frequency_churn.png", bbox_inches='tight')
plt.close()


# 5. Last Purchase Recency vs Churn (Box Plot)

plt.figure(figsize=(10, 6))
sns.boxplot(
    x='Churned', 
    y='Last_Purchase_Days_Ago', 
    data=df, 
    hue='Churned',  # Added hue parameter
    palette=['#4c72b0', '#c44e52'],
    legend=False  # Disables redundant legend
)
plt.title("Last Purchase Recency vs Churn Status", fontsize=14, pad=20)
plt.xlabel("Churned")
plt.ylabel("Days Since Last Purchase")
plt.savefig("visualizations/recency_churn.png", bbox_inches='tight')
plt.close()


# 6. Churn Heatmap (Age Group vs Country)

df['Age_Group'] = pd.cut(df['Age'], 
                         bins=[18, 25, 35, 45, 55, 65],
                         labels=['18-25', '26-35', '36-45', '46-55', '56-55'])

# Pivot table for heatmap - Add observed=False
heatmap_data = df.pivot_table(
    index='Country', 
    columns='Age_Group', 
    values='Churned', 
    aggfunc=lambda x: (x == 'Yes').mean(),
    observed=False  # Explicitly set observed parameter
)
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".0%", cmap="YlGnBu", linewidths=.5)
plt.title("Churn Rate by Age Group and Country", fontsize=14, pad=20)
plt.xlabel("Age Group")
plt.ylabel("Country")
plt.savefig("visualizations/churn_heatmap.png", bbox_inches='tight')
plt.close()
