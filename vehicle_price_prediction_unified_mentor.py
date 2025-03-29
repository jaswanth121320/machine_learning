#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
file_path = r"C:\Users\Jayadeep\OneDrive\Desktop\Unified_mentor\Vehicle Price Prediction\dataset.csv"
df = pd.read_csv(file_path)

# Drop irrelevant columns (text-based and not useful for numerical predictions)
df.drop(columns=['name', 'description'], inplace=True, errors='ignore')

# Remove rows where price is missing (target variable)
df.dropna(subset=['price'], inplace=True)

# Separate target variable
y = df['price']
X = df.drop(columns=['price'])

# Identify numerical and categorical features
num_features = ['year', 'cylinders', 'mileage', 'doors']
cat_features = ['make', 'model', 'fuel', 'transmission', 'trim', 'body', 'exterior_color', 'interior_color', 'drivetrain']

# Define data preprocessing steps
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Fill missing numerical values with mean
    ('scaler', StandardScaler())  # Scale numerical features
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing categorical values with mode
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-Hot Encoding for categorical data
])

# Combine numerical & categorical transformations
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# Split dataset into training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge Regression Model with Hyperparameter Tuning
ridge_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('ridge', Ridge())
])

ridge_params = {'ridge__alpha': [0.1, 1, 10, 100]}  # Ridge regularization parameters

ridge_grid = GridSearchCV(ridge_pipeline, ridge_params, cv=5, scoring='r2', n_jobs=-1, verbose=1)
ridge_grid.fit(X_train, y_train)

# Decision Tree Regressor with Hyperparameter Tuning
dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('dt', DecisionTreeRegressor())
])

dt_params = {
    'dt__max_depth': [5, 10, 15, None],
    'dt__min_samples_split': [2, 5, 10],
    'dt__min_samples_leaf': [1, 2, 5]
}

dt_grid = GridSearchCV(dt_pipeline, dt_params, cv=5, scoring='r2', n_jobs=-1, verbose=1)
dt_grid.fit(X_train, y_train)

# Evaluate Models
models = {'Ridge Regression': ridge_grid, 'Decision Tree': dt_grid}

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n📊 {name} Results:")
    print(f"✅ MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"✅ RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
    print(f"✅ R² Score: {r2_score(y_test, y_pred):.2f}\n")

# Cross-Validation Scores
ridge_cv_scores = cross_val_score(ridge_grid.best_estimator_, X_train, y_train, cv=5, scoring='r2')
dt_cv_scores = cross_val_score(dt_grid.best_estimator_, X_train, y_train, cv=5, scoring='r2')

# Visualizing Cross-Validation Results
plt.figure(figsize=(10, 5))
plt.plot(range(1, 6), ridge_cv_scores, marker='o', label='Ridge Regression', color='blue')
plt.plot(range(1, 6), dt_cv_scores, marker='s', label='Decision Tree', color='red')
plt.axhline(y=np.mean(ridge_cv_scores), color='blue', linestyle='dashed', label='Ridge Avg Score')
plt.axhline(y=np.mean(dt_cv_scores), color='red', linestyle='dashed', label='DT Avg Score')
plt.xlabel("Cross-Validation Fold")
plt.ylabel("R² Score")
plt.title("Cross-Validation Results for Ridge & Decision Tree")
plt.legend()
plt.show()

# 🔹 Fixed: Manual Input Prediction
def manual_prediction(model):
    print("\n⚡ Enter values for manual prediction (one by one):")
    user_input = {}

    for col in num_features:
        user_input[col] = float(input(f"Enter {col}: "))

    for col in cat_features:
        user_input[col] = input(f"Enter {col}: ")

    # Convert input to DataFrame
    input_df = pd.DataFrame([user_input])

    # 🔹 FIX: Ensure preprocessor is fitted before transforming
    fitted_preprocessor = model.named_steps['preprocessor']
    input_transformed = fitted_preprocessor.transform(input_df)

    # Predict price
    pred = model.named_steps['ridge'].predict(input_transformed)[0] if 'ridge' in model.named_steps else model.named_steps['dt'].predict(input_transformed)[0]
    print(f"\n🚗 Predicted Vehicle Price: ${pred:,.2f}")

# Ask user which model to use
print("\n🛠️ Choose a model for prediction:")
choice = input("Type 'ridge' for Ridge Regression or 'tree' for Decision Tree: ").strip().lower()

if choice == 'ridge':
    manual_prediction(ridge_grid.best_estimator_)
elif choice == 'tree':
    manual_prediction(dt_grid.best_estimator_)
else:
    print("❌ Invalid choice! Exiting.")


# In[ ]:




