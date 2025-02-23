

# Churn Prediction Model (Logistic Regression vs. Decision Tree)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Load and Prepare Data

df = pd.read_excel("Cleaned_Customer_Retention_Data.xlsx")  # Update path if needed

# Encode target variable
df['Churned'] = df['Churned'].map({'Yes': 1, 'No': 0})

# Define features
categorical_features = ['Country', 'Gender']
numerical_features = ['Age', 'Purchase_Frequency', 'Avg_Purchase_Value', 'Last_Purchase_Days_Ago']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ])

X = preprocessor.fit_transform(df.drop('Churned', axis=1))
y = df['Churned']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 2. Logistic Regression Model

lr = LogisticRegression(class_weight='balanced', max_iter=1000)
lr.fit(X_train, y_train)

# Predictions
y_pred_lr = lr.predict(X_test)
y_proba_lr = lr.predict_proba(X_test)[:, 1]

# Metrics
print("="*50)
print("Logistic Regression Performance:")
print("="*50)
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_lr):.2f}")
print(f"F1-Score: {f1_score(y_test, y_pred_lr):.2f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_lr):.2f}\n")

# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
ConfusionMatrixDisplay(cm_lr).plot(cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.show()


# 3. Decision Tree Model

dt = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42
)
dt.fit(X_train, y_train)

# Predictions
y_pred_dt = dt.predict(X_test)
y_proba_dt = dt.predict_proba(X_test)[:, 1]

# Metrics
print("="*50)
print("Decision Tree Performance:")
print("="*50)
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_dt):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_dt):.2f}")
print(f"F1-Score: {f1_score(y_test, y_pred_dt):.2f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_dt):.2f}\n")

# Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
ConfusionMatrixDisplay(cm_dt).plot(cmap='Greens')
plt.title("Decision Tree Confusion Matrix")
plt.show()


# 4. Feature Importance Analysis

# Get feature names
cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
num_features = numerical_features
all_features = np.concatenate([cat_features, num_features])

# Plot Decision Tree feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x=dt.feature_importances_, y=all_features, palette="viridis")
plt.title("Decision Tree Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# 5. Model Comparison Report

print("="*50)
print("Final Model Comparison:")
print("="*50)
print(f"{'Metric':<15} {'Logistic Regression':<20} {'Decision Tree':<15}")
print(f"{'Accuracy':<15} {accuracy_score(y_test, y_pred_lr):.2f}{'':<18} {accuracy_score(y_test, y_pred_dt):.2f}")
print(f"{'Precision':<15} {precision_score(y_test, y_pred_lr):.2f}{'':<18} {precision_score(y_test, y_pred_dt):.2f}")
print(f"{'Recall':<15} {recall_score(y_test, y_pred_lr):.2f}{'':<18} {recall_score(y_test, y_pred_dt):.2f}")
print(f"{'F1-Score':<15} {f1_score(y_test, y_pred_lr):.2f}{'':<18} {f1_score(y_test, y_pred_dt):.2f}")
print(f"{'ROC-AUC':<15} {roc_auc_score(y_test, y_proba_lr):.2f}{'':<18} {roc_auc_score(y_test, y_proba_dt):.2f}")




#### OUTPUT ####

# ==================================================
# Final Model Comparison:
# ==================================================
# Metric          Logistic Regression  Decision Tree
# Accuracy        0.54                   0.53
# Precision       0.54                   0.54
# Recall          0.61                   0.36
# F1-Score        0.57                   0.43
# ROC-AUC         0.52                   0.52