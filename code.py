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

# ========== 1. LOAD DATA WITH ENCODING & FILE CHECK ==========
import os
if not os.path.isfile("AviationData.csv"):
    raise FileNotFoundError("File 'AviationData.csv' not found in current directory.")

df = pd.read_csv("AviationData.csv", encoding="latin1")
print(df.head())
print(df.info())
print(df['Aircraft.damage'].value_counts())

# ========== 2. PREPROCESSING ==========
drop_cols = [
    'Event.Id', 'Accident.Number', 'Publication.Date', 'Report.Status',
    'Registration.Number', 'Event.Date', 'Location', 'Country',
    'Airport.Code', 'Airport.Name'
]
df = df.drop(columns=drop_cols, errors='ignore')
df = df.dropna(subset=['Aircraft.damage'])
df = df.fillna('Unknown')

# Label encode all categorical columns except the target
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
if 'Aircraft.damage' in categorical_columns:
    categorical_columns.remove('Aircraft.damage')

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Label encode the target
le_damage = LabelEncoder()
df['Aircraft.damage'] = le_damage.fit_transform(df['Aircraft.damage'].astype(str))

# Ensure no missing or non-numeric types
assert not df.isnull().any().any(), "Missing values present after preprocessing."
assert all([np.issubdtype(df[col].dtype, np.number) for col in df.columns]), "Non-numeric data found after encoding."

# ========== 3. TRAIN-TEST SPLIT ==========
X = df.drop('Aircraft.damage', axis=1)
y = df['Aircraft.damage']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ========== 4. BALANCE THE CLASSES (SMOTE) ==========
# After all encoding and imputing
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

assert X_train.select_dtypes(include=['object']).empty, "Non-numeric cols remain"
assert not X_train.isnull().values.any(), "NaNs in X_train"
assert not y_train.isnull().values.any(), "NaNs in y_train"
assert X_train.shape[0] == y_train.shape[0], "X_train/y_train size mismatch"

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ========== 5. SCALING ==========
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# ========== 6. RANDOM FOREST CLASSIFIER ==========
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_res_scaled, y_train_res)

# ========== 7. PREDICTION AND EVALUATION ==========
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

print('Accuracy:', accuracy)
print('F1 Score:', f1)
print('Recall:', recall)
print('Precision:', precision)
print('Confusion Matrix:\n', cm)
print(classification_report(y_test, y_pred, target_names=le_damage.classes_))

# ========== 8. VISUALIZE CONFUSION MATRIX ==========
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_damage.classes_, yticklabels=le_damage.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# ========== 9. SAVE THE MODEL AND ENCODERS ==========
joblib.dump(model, 'aviation_damage_rf.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_damage, 'label_encoder_damage.pkl')
joblib.dump(label_encoders, 'feature_label_encoders.pkl')

print("Model, scaler, and label encoders saved successfully!")
