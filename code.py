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
# Correct path formatting for Windows. Use an 'r' before the string, or double backslashes, or forward slashes.
df = pd.read_csv("AviationData.csv")
print(df.head())
print(df.info())
print(df['Aircraft.damage'].value_counts())
drop_cols = [
    'Event.Id', 'Accident.Number', 'Publication.Date', 'Report.Status',
    'Registration.Number', 'Event.Date', 'Location', 'Country',
    'Airport.Code', 'Airport.Name'
]
df = df.drop(columns=drop_cols, errors='ignore')
df = df.dropna(subset=['Aircraft.damage'])
df = df.fillna('Unknown')

categorical_columns = df.select_dtypes(include='object').columns.tolist()
if 'Aircraft.damage' in categorical_columns:
    categorical_columns.remove('Aircraft.damage')

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

le_damage = LabelEncoder()
df['Aircraft.damage'] = le_damage.fit_transform(df['Aircraft.damage'].astype(str))
X = df.drop('Aircraft.damage', axis=1)
y = df['Aircraft.damage']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)
model = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight='balanced'
)
model.fit(X_train_res_scaled, y_train_res)
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
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_damage.classes_, yticklabels=le_damage.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
joblib.dump(model, 'aviation_damage_rf.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_damage, 'label_encoder_damage.pkl')

print("Model and encoders saved successfully!")
