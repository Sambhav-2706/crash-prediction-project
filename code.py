import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def main():

    # ============ 1. LOAD DATA (Update path as needed) ============
    file_path = "AviationData.csv"
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found. Please check the path.")

    # Specify encoding to avoid Unicode errors (adjust if needed)
    df = pd.read_csv(file_path, encoding="latin1")

    print("Initial Data Snapshot:")
    print(df.head())
    print(df.info())
    print(df['Aircraft.damage'].value_counts())

    # ============ 2. PREPROCESSING ============
    # Drop irrelevant columns - adjust as per your dataset
    drop_cols = [
        'Event.Id', 'Accident.Number', 'Publication.Date', 'Report.Status',
        'Registration.Number', 'Event.Date', 'Location', 'Country',
        'Airport.Code', 'Airport.Name'
    ]
    df.drop(columns=drop_cols, errors='ignore', inplace=True)

    # Drop rows with missing target and fill other NAs with 'Unknown'
    df.dropna(subset=['Aircraft.damage'], inplace=True)
    df.fillna('Unknown', inplace=True)

    # Label encode categorical features (excluding target)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'Aircraft.damage' in categorical_cols:
        categorical_cols.remove('Aircraft.damage')

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Encode the target variable
    le_target = LabelEncoder()
    df['Aircraft.damage'] = le_target.fit_transform(df['Aircraft.damage'].astype(str))

    # Ensure no missing values remain
    if df.isnull().values.any():
        raise ValueError("Missing values exist after preprocessing. Please check data cleaning steps.")

    # Ensure all columns are numeric
    non_numeric = df.select_dtypes(include=['object']).columns.tolist()
    if non_numeric:
        raise TypeError(f"Non-numeric columns remain after encoding: {non_numeric}")

    # ============ 3. SPLIT DATA ============
    X = df.drop('Aircraft.damage', axis=1)
    y = df['Aircraft.damage']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Reset indices to ensure alignment
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # ============ 4. CHECK & CLEAN TRAINING DATA BEFORE SMOTE ============
    # Check for any non-numeric columns (should not happen here)
    assert X_train.select_dtypes(include=['object']).empty, "Non-numeric columns exist in X_train"
    assert not X_train.isnull().values.any(), "Missing values in X_train"
    assert not y_train.isnull().values.any(), "Missing values in y_train"
    assert X_train.shape[0] == y_train.shape[0], "Mismatch in X_train and y_train rows"

    # ============ 5. APPLY SMOTE TO BALANCE CLASSES ============
    # Clean indices
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    # Clean types, fill NAs, and force all numeric
    for col in X_train.columns:
        if X_train[col].dtype == 'object' or str(X_train[col].dtype).startswith('category'):
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))

    X_train = X_train.fillna(-999)
    y_train = y_train.fillna(-999)

    # Final type check and assertion
    assert X_train.select_dtypes(include=['object', 'category', 'string']).empty, "Non-numeric X columns"
    assert not X_train.isnull().values.any(), "NaNs in X_train"
    assert not y_train.isnull().values.any(), "NaNs in y_train"
    assert X_train.shape[0] == y_train.shape[0], "Mismatched train sizes"

    # Now apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # ============ 6. FEATURE SCALING ============
    scaler = StandardScaler()
    X_train_res_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    # ============ 7. TRAIN MODEL ============
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_res_scaled, y_train_res)

    # ============ 8. EVALUATE MODEL ============
    y_pred = model.predict(X_test_scaled)

    print("Model Performance Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))

    # ============ 9. VISUALIZE CONFUSION MATRIX ============
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le_target.classes_, yticklabels=le_target.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # ============ 10. SAVE MODEL AND ENCODERS ============
    joblib.dump(model, "aviation_damage_rf.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(le_target, "label_encoder_damage.pkl")
    joblib.dump(label_encoders, "feature_label_encoders.pkl")

    print("Model, scaler, and encoders saved successfully.")

if __name__ == "__main__":
    main()
