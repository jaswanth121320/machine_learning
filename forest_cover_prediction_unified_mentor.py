#!/usr/bin/env python
# coding: utf-8

# 

# In[6]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = r"C:\Users\Jayadeep\OneDrive\Desktop\Unified_mentor\forest_cover_prediction\train.csv"
df = pd.read_csv(file_path)

# Drop 'Id' column since it's not a feature
df.drop(columns=['Id'], inplace=True)

# Shift Cover_Type to start from 0 (XGBoost requires zero-based classes)
df['Cover_Type'] = df['Cover_Type'] - 1

# Split features and target
X = df.drop(columns=['Cover_Type'])  # Features
y = df['Cover_Type']  # Target variable

# Identify non-binary columns for scaling
non_binary_cols = [col for col in X.columns if len(X[col].unique()) > 2]

# Apply feature scaling only to non-binary columns
scaler = StandardScaler()
X[non_binary_cols] = scaler.fit_transform(X[non_binary_cols])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize models
rf = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Optimized hyperparameter tuning using GridSearchCV
param_grid_rf = {
    'n_estimators': [100, 150],  # Reduced size for faster execution
    'max_depth': [10, 15],  # Reduced search space
}

param_grid_xgb = {
    'n_estimators': [100, 150],
    'max_depth': [6, 10],
}

# Run GridSearchCV with verbose=1 for progress updates
print("\n🔍 Running GridSearch for Random Forest...")
grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_rf.fit(X_train, y_train)

print("\n🔍 Running GridSearch for XGBoost...")
grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_xgb.fit(X_train, y_train)

# Best models
best_rf = grid_rf.best_estimator_
best_xgb = grid_xgb.best_estimator_

# Predictions
rf_preds = best_rf.predict(X_test)
xgb_preds = best_xgb.predict(X_test)

# Evaluation
print("\n✅ Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
print("\n✅ XGBoost Accuracy:", accuracy_score(y_test, xgb_preds))

print("\n📊 Random Forest Classification Report:\n", classification_report(y_test, rf_preds))
print("\n📊 XGBoost Classification Report:\n", classification_report(y_test, xgb_preds))

# Confusion Matrix Visualization
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt="d", cmap="Blues", ax=ax[0])
ax[0].set_title("Random Forest Confusion Matrix")

sns.heatmap(confusion_matrix(y_test, xgb_preds), annot=True, fmt="d", cmap="Reds", ax=ax[1])
ax[1].set_title("XGBoost Confusion Matrix")

plt.show()

# Manual Input Prediction
def manual_prediction(model):
    print("\n⚡ Enter values for manual prediction (one by one):")
    user_input = []
    for col in X.columns:
        value = float(input(f"Enter {col}: "))  # Ensure numeric input
        user_input.append(value)
    
    # Convert to DataFrame & scale non-binary features
    input_df = pd.DataFrame([user_input], columns=X.columns)
    input_df[non_binary_cols] = scaler.transform(input_df[non_binary_cols])
    
    # Predict
    pred = model.predict(input_df)[0]
    print(f"\n🌲 Predicted Forest Cover Type: {pred + 1}")  # Convert back to original label

# Ask user which model to use
print("\n🛠️ Choose a model for prediction:")
choice = input("Type 'rf' for Random Forest or 'xgb' for XGBoost: ").strip().lower()
if choice == 'rf':
    manual_prediction(best_rf)
elif choice == 'xgb':
    manual_prediction(best_xgb)
else:
    print("❌ Invalid choice! Exiting.")


# In[ ]:




