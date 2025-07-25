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
from collections import Counter
import os

def main():
    # ============== 1. LOAD DATA ==============
    file_path = "AviationData.csv"
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found in the current directory!")

    # Load CSV with encoding that avoids errors
    df = pd.read_csv(file_path, encoding="latin1")

    print("Data loaded. Sample:")
    print(df.head())
    print(df.info())
    print("Target distribution:")
    print(df['Aircraft.damage'].value_counts())

    # ============== 2. PREPROCESSING ==============
    drop_cols = [
        'Event.Id', 'Accident.Number', 'Publication.Date', 'Report.Status',
        'Registration.Number', 'Event.Date', 'Location', 'Country',
        'Airport.Code', 'Airport.Name'
    ]
    df.drop(columns=drop_cols, errors='ignore', inplace=True)

    # Remove rows with missing target and fill other missing values
    df.dropna(subset=['Aircraft.damage'], inplace=True)
    df.fillna('Unknown', inplace=True)

    # Encode categorical features (except the target)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'Aircraft.damage' in categorical_cols:
        categorical_cols.remove('Aircraft.damage')

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Encode target
    le_target = LabelEncoder()
    df['Aircraft.damage'] = le_target.fit_transform(df['Aircraft.damage'].astype(str))

    # Assert no missing values remain
    if df.isnull().values.any():
        raise ValueError("Missing values present after preprocessing! Please check data cleaning.")

    # Assert all data is numeric
    non_numeric_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if len(non_numeric_cols) > 0:
        raise TypeError(f"Non-numeric columns remain: {non_numeric_cols}")

    # ============== 3. SPLIT DATA ==============
    X = df.drop('Aircraft.damage', axis=1)
    y = df['Aircraft.damage']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Reset indices after split
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # ============== 4. HANDLE RARE CLASSES FOR SMOTE ==============
    counts = Counter(y_train)
    print("Class frequency before rare class handling:", counts)
    min_samples = 6  # SMOTE default requires at least k_neighbors+1 samples per class
    rare_classes = [cls for cls, count in counts.items() if count < min_samples]

    if rare_classes:
        print(f"Found rare classes with fewer than {min_samples} samples: {rare_classes}, dropping them.")
        mask = y_train.isin([cls for cls in counts if counts[cls] >= min_samples])
        X_train = X_train.loc[mask].reset_index(drop=True)
        y_train = y_train.loc[mask].reset_index(drop=True)

        # Confirm after removal
        counts_after = Counter(y_train)
        print("Class frequency after removing rare classes:", counts_after)

    # ============== 5. FINAL DATA CHECK BEFORE SMOTE ==============

    non_numeric_cols = X_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    print("Non-numeric columns before SMOTE:", non_numeric_cols)
    print("Any NaNs in X_train?", X_train.isnull().values.any())
    print("Any NaNs in y_train?", y_train.isnull().values.any())
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    assert len(non_numeric_cols) == 0, f"Non-numeric columns remain: {non_numeric_cols}"
    assert not X_train.isnull().values.any(), "Missing values present in X_train"
    assert not y_train.isnull().values.any(), "Missing values present in y_train"
    assert X_train.shape[0] == y_train.shape[0], "Number of rows in X_train and y_train do not match"

    # ============== 6. APPLY SMOTE ==============
    print("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print("SMOTE applied successfully.")

    # ============== 7. FEATURE SCALING ==============
    scaler = StandardScaler()
    X_train_res_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    # ============== 8. MODEL TRAINING ==============
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_res_scaled, y_train_res)

    # ============== 9. MODEL EVALUATION ==============
    y_pred = model.predict(X_test_scaled)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))

    # Plot confusion matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=le_target.classes_, yticklabels=le_target.classes_
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # ============== 10. SAVE MODEL AND ENCODERS ==============
    joblib.dump(model, "aviation_damage_rf.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(le_target, "label_encoder_damage.pkl")
    joblib.dump(label_encoders, "feature_label_encoders.pkl")

    print("Saved model, scaler, and label encoders successfully.")

if __name__ == "__main__":
    main()
