import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# Extract data
dataset = pd.read_csv("/Users/williamvm/Documents/Projects/Fraud Detection/creditcard.csv")

# Check for missing values
# print(dataset.isnull().sum()) 
# No missing values

# Feature Engineering
dataset['log_amount'] = np.log1p(dataset['Amount']) # Applies log transformation to normalize data points
dataset.drop(columns=['Amount'], inplace=True) # Drop original column

dataset['hour'] = (dataset['Time'] // 3600) % 24 # Convert time to hours of day
dataset.drop(columns=['Time'], inplace=True) # Drop original columns

# Handle class imbalance
# Only 0.17% of the transactions are fraud -> Highly imbalanced dataset
x = dataset.drop(columns=['Class']) # Store features
y = dataset['Class'] # Store class

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

# Build XGBoost Model
model = XGBClassifier(scale_pos_weight=10, n_estimators=600, max_depth=6, learning_rate=0.1, random_state=42)

# Train
model.fit(x_train, y_train)

# Predict on the test set
y_pred = model.predict(x_test)

# Classification report result
print(classification_report(y_test, y_pred))

# Accuracy result
print("Test Accuracy: ", accuracy_score(y_test, y_pred))