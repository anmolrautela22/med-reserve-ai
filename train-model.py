# train_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

print("⏳ Generating clinical dataset...")
np.random.seed(42)
num_samples = 2000

# Generate realistic mock parameters
age = np.random.randint(18, 90, size=num_samples)
spo2 = np.random.randint(80, 100, size=num_samples)
heart_rate = np.random.randint(50, 140, size=num_samples)
comorbidities = np.random.choice([0, 1, 2, 3], size=num_samples, p=[0.4, 0.3, 0.2, 0.1])

# Create a risk score calculation to act as ground truth rules for our ML model
# (Lower SpO2 + Older Age + Comorbidities = Higher Risk)
risk_score = (100 - spo2) * 1.5 + (age * 0.2) + (comorbidities * 5)

# Assign Triage Level: 0=Low Risk (General Ward), 1=Medium Risk (High Dependency Unit), 2=High Risk (ICU)
triage_level = np.where(risk_score > 35, 2, np.where(risk_score > 20, 1, 0))

# Estimate length of stay based loosely on triage level
length_of_stay = np.random.randint(1, 5, size=num_samples) + (triage_level * 5)

# Compile DataFrame
df = pd.DataFrame({
    'Age': age,
    'SpO2': spo2,
    'Heart_Rate': heart_rate,
    'Comorbidities': comorbidities,
    'Triage_Level': triage_level,
    'LOS': length_of_stay
})

# Split features and label (We will predict the Triage Level)
X = df[['Age', 'SpO2', 'Heart_Rate', 'Comorbidities']]
y = df['Triage_Level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("⚡ Training Machine Learning Pipeline...")
# Pipeline chains feature scaling and the ML model together safely
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)

# Quick validation check
y_pred = pipeline.predict(X_test)
print("\n✅ Model Trained Successfully! Evaluation Report:")
print(classification_report(y_test, y_pred))

# Save the trained pipeline as a local binary artifact file
with open('med_reserve_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print("💾 Model artifact successfully saved to 'med_reserve_model.pkl'")