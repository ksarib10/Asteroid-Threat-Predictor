#!/usr/bin/env python3
# --------------------------------------------------------------
# Asteroid Threat Predictor - Machine Learning Pipeline
# Author: Sarib Khan
# --------------------------------------------------------------

import pandas as pd
import numpy as np
import joblib
import warnings

from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# --------------------------------------------------------------
# Step 1: Load Data and Understand it
# --------------------------------------------------------------
print("Loading data...")

data = pd.read_csv("C:\\Users\\mkkha\\OneDrive\\Desktop\\AI SEM PROJ\\Data\\Nasa_Dataset.csv")

print(f"Dataset shape: {data.shape}")
print(f"Columns: {list(data.columns)}\n")
print(data.head())
print(data.tail())
print(data.shape)
print(data.info())
print(data.describe())


# --------------------------------------------------------------
# Step 2: Data Cleaning
# --------------------------------------------------------------
print("Cleaning data...")

# Drop identifier columns
columns_to_drop = ["neo_id", "name", "orbiting_body"]
data.drop(columns=columns_to_drop, inplace=True)
print(f"Dropped columns: {columns_to_drop}")

# Fill missing values with median
cols_with_missing = ["absolute_magnitude", "estimated_diameter_min", "estimated_diameter_max"]
for col in cols_with_missing:
    median_value = data[col].median()
    data[col].fillna(median_value, inplace=True)
    print(f"Filled missing values in {col} with median: {median_value}")

# Encode target variable
label_encoder = LabelEncoder()
data["is_hazardous"] = label_encoder.fit_transform(data["is_hazardous"])
print("Encoded target variable 'is_hazardous' (0 = not hazardous, 1 = hazardous)\n")

# --------------------------------------------------------------
# Step 3: Feature Engineering
# --------------------------------------------------------------
print("Engineering new features...")

# Create additional features
data["estimated_diameter_avg"] = (data["estimated_diameter_min"] + data["estimated_diameter_max"]) / 2
data["diameter_range"] = data["estimated_diameter_max"] - data["estimated_diameter_min"]
data["velocity_distance_ratio"] = data["relative_velocity"] / data["miss_distance"]

# Log transformations
data["log_miss_distance"] = np.log1p(data["miss_distance"])
data["log_relative_velocity"] = np.log1p(data["relative_velocity"])

# Drop old columns no longer needed
drop_features = ["estimated_diameter_min", "estimated_diameter_max", "relative_velocity", "miss_distance"]
data.drop(columns=drop_features, inplace=True)
print(f"Dropped redundant columns: {drop_features}\n")

# --------------------------------------------------------------
# Step 4: Feature Selection
# --------------------------------------------------------------
print("Selecting best features...")

X = data.drop("is_hazardous", axis=1)
y = data["is_hazardous"]

selector = SelectKBest(score_func=f_classif, k=6)
X_selected = selector.fit_transform(X, y)
selected_columns = X.columns[selector.get_support()]

# Convert selected features back to DataFrame
X = pd.DataFrame(X_selected, columns=selected_columns)

print(f"Selected features: {list(selected_columns)}\n")

# --------------------------------------------------------------
# Step 5: Scaling
# --------------------------------------------------------------

print("Scaling features with QuantileTransformer...")

transformer = QuantileTransformer(output_distribution='normal', random_state=42)
X_scaled = transformer.fit_transform(X)

print("Scaling complete.\n")

# --------------------------------------------------------------
# Step 6: Apply SMOTE
# --------------------------------------------------------------

print("Applying SMOTE to balance classes...")

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print(f"Class distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())
print()

# --------------------------------------------------------------
# Step 7: Train-Test Split
# --------------------------------------------------------------

print("Splitting data into training and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.3,
    random_state=42,
    stratify=y_resampled
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}\n")

# --------------------------------------------------------------
# Step 8: Model Training
# --------------------------------------------------------------

print("Training Random Forest model...")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("Model training complete.\n")

# --------------------------------------------------------------
# Step 9: Model Evaluation
# --------------------------------------------------------------
print("Evaluating model...")

y_pred = rf_model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}\n")

# --------------------------------------------------------------
# Step 10: Save Model and Transformer
# --------------------------------------------------------------
print("Saving model and transformer...")

joblib.dump(rf_model, "models/asteroid_rf_model.pkl")
joblib.dump(transformer, "models/transformer.pkl")

print("--> Model and transformer saved successfully!")
print("Pipeline complete.")
