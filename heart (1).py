#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset (Replace with actual file path)
data = pd.read_csv(r"C:\Users\naris\OneDrive\Desktop\dataset.csv")


# Display first few rows
print("Dataset Preview:\n", data.head())

# Define features and target variable
X = data.drop(columns=['target'])
y = data['target']

# Split data into training/testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model and scaler
joblib.dump(model, "heart_disease_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# --- Graph 1: Feature Importance ---
plt.figure(figsize=(10, 5))
feature_importance = model.feature_importances_
features = X.columns
sns.barplot(x=feature_importance, y=features, palette="viridis")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Heart Disease Prediction")
plt.show()

# --- Graph 2: Heart Disease Distribution ---
plt.figure(figsize=(6, 4))
sns.countplot(x=y, palette="coolwarm")
plt.xlabel("Heart Disease Status")
plt.ylabel("Number of Patients")
plt.title("Distribution of Heart Disease Cases")
plt.xticks(ticks=[0, 1], labels=["No Disease", "Heart Disease"])
plt.show()

# --- Graph 3: Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Heart Disease"], yticklabels=["No Disease", "Heart Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Heart Disease Prediction")
plt.show()

# --- Prediction Function ---
def predict_heart_disease():
    print("\nEnter Patient Details:")
    age = float(input("Age: "))
    sex = int(input("Sex (0 = Female, 1 = Male): "))
    cp = int(input("Chest Pain Type (1-4): "))
    trestbps = float(input("Resting Blood Pressure: "))
    chol = float(input("Serum Cholesterol: "))
    fbs = int(input("Fasting Blood Sugar (0 = <120mg/dL, 1 = >120mg/dL): "))
    restecg = int(input("Resting ECG Results (0-2): "))
    thalach = float(input("Max Heart Rate Achieved: "))
    exang = int(input("Exercise Induced Angina (0 = No, 1 = Yes): "))
    oldpeak = float(input("ST Depression Induced by Exercise: "))
    slope = int(input("Slope of Peak Exercise ST Segment (1-3): "))

    # Convert input into a DataFrame
    patient_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]])
    patient_data = scaler.transform(patient_data)  # Scale input data

    # Predict heart disease
    prediction = model.predict(patient_data)
    result = "Heart Disease Detected" if prediction[0] == 1 else "Normal Heart Condition"
    print("\nPrediction:", result)

# Run prediction function
predict_heart_disease()


# In[ ]:




