
import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─── Load Data ───────────────────────────────────────────────────────────────
data_path = os.path.join(os.path.dirname(__file__), "survey.csv")
df = pd.read_csv(data_path)

# ─── Drop Unnecessary Columns ────────────────────────────────────────────────
df.drop(columns=["Timestamp", "state","comments"], inplace=True, errors='ignore')

# ─── Drop rows with missing target ───────────────────────────────────────────
df = df[df['treatment'].notna()]

# ─── Handle Outliers in Age ──────────────────────────────────────────────────
df = df[(df['Age'] > 15) & (df['Age'] < 100)]

# ─── Split Features and Target ───────────────────────────────────────────────
y = df['treatment']
X = df.drop('treatment', axis=1)

# ─── Identify Column Types ───────────────────────────────────────────────────
categorical_columns = X.select_dtypes(include='object').columns.tolist()
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# ─── Save LabelEncoders for Categorical Inputs ───────────────────────────────
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# ─── Label Encode Target ─────────────────────────────────────────────────────
target_le = LabelEncoder()
y_enc = target_le.fit_transform(y)
label_encoders["__target__"] = target_le  # Save target encoder for later use if needed

# ─── Train-Test Split ────────────────────────────────────────────────────────
X_train, X_test, y_train_enc, y_test_enc = train_test_split(X, y_enc, test_size=0.2, random_state=42)

# ─── Pipelines for Preprocessing ─────────────────────────────────────────────
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_columns),
    ('cat', categorical_pipeline, categorical_columns)
])

# ─── Build Model Pipeline ────────────────────────────────────────────────────
clf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# ─── Fit Model ───────────────────────────────────────────────────────────────
clf_pipeline.fit(X_train, y_train_enc)

# ─── Predictions ─────────────────────────────────────────────────────────────
y_pred = clf_pipeline.predict(X_test)

# ─── Evaluation ──────────────────────────────────────────────────────────────
print("\n✅ Model Trained Successfully!")
print(f"Training Score: {clf_pipeline.score(X_train, y_train_enc):.4f}")
print(f"Testing Score: {clf_pipeline.score(X_test, y_test_enc):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_enc, y_pred))
print("\nClassification Report:")
print(classification_report(y_test_enc, y_pred))
scores = cross_val_score(clf_pipeline, X, y_enc, cv=5)
print(f"\nCross-Validation Scores: {scores}")
print(f"CV Accuracy: {scores.mean():.4f}")

# ─── Save Model and Label Encoders ───────────────────────────────────────────
model_path = os.path.join(os.path.dirname(__file__), "mental_health_model.pkl")
le_path = os.path.join(os.path.dirname(__file__), "label_encoders.pkl")

with open(model_path, 'wb') as f:
    pickle.dump(clf_pipeline, f)

with open(le_path, 'wb') as f:
    pickle.dump(label_encoders, f)

print(f"\n🔒 Model saved to: {model_path}")
print(f"🔒 Label encoders saved to: {le_path}")
