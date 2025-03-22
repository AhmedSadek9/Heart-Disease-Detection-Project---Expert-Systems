import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load your dataset
df = pd.read_csv(r"d:/Semester Four/Intelligent Programming/Project 1/heart_cleaned_data.csv")

# Preprocess the data if needed (e.g., one-hot encoding)
df_encoded = pd.get_dummies(df.drop('target', axis=1))
target = df['target']

# Save feature columns to match input later
joblib.dump(df_encoded.columns.tolist(), 'model_features.joblib')

X_train, X_test, y_train, y_test = train_test_split(df_encoded, target, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'heart_disease_decision_tree_model.joblib')
print("✅ Model saved as heart_disease_decision_tree_model.joblib")
