import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

def get_data():
    return pd.read_csv("AviationData.csv", encoding="latin1")

def preprocess(df):
    drop_cols = ['Event.Id', 'Accident.Number', 'Publication.Date', 'Report.Status',
                 'Registration.Number', 'Event.Date', 'Location', 'Country',
                 'Airport.Code', 'Airport.Name']
    df.drop(columns=drop_cols, errors='ignore', inplace=True)
    df.dropna(subset=['Aircraft.damage'], inplace=True)
    df.fillna('Unknown', inplace=True)

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'Aircraft.damage' in cat_cols:
        cat_cols.remove('Aircraft.damage')
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    le_target = LabelEncoder()
    df['Aircraft.damage'] = le_target.fit_transform(df['Aircraft.damage'].astype(str))
    return df, label_encoders, le_target

def check_data(X_train, y_train):
    non_numeric_cols = X_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    st.write("Non-numeric columns (should be empty):", non_numeric_cols)
    st.write("Any NaNs in X_train?", X_train.isnull().values.any())
    st.write("Any NaNs in y_train?", y_train.isnull().values.any())
    st.write("X_train shape:", X_train.shape)
    st.write("y_train shape:", y_train.shape)

    assert len(non_numeric_cols) == 0, f"Non-numeric cols remain: {non_numeric_cols}"
    assert not X_train.isnull().any().any(), "Missing values in X_train"
    assert not y_train.isnull().any(), "Missing values in y_train"
    assert X_train.shape[0] == y_train.shape[0], "X_train and y_train count mismatch"

# Streamlit UI
st.title("Aviation Damage Model Training")
df = get_data()
df, label_encoders, le_target = preprocess(df)

X = df.drop('Aircraft.damage', axis=1)
y = df['Aircraft.damage']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

# Re-encode if needed
for col in X_train.select_dtypes(include=['object', 'category', 'string']).columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))

X_train = X_train.fillna(-999)
y_train = y_train.fillna(-999)

# Diagnostic display
check_data(X_train, y_train)

# Now safely apply SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_res_scaled, y_train_res)

y_pred = model.predict(X_test_scaled)

st.header("Evaluation Metrics")
st.write("Accuracy:", accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix:")
st.write(cm)
st.text(classification_report(y_test, y_pred, target_names=le_target.classes_))
