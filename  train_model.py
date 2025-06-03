import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# === Step 1: Load Dataset ===
df = pd.read_csv('test.csv')

# === Step 2: Drop ID-like columns ===
df.drop(columns=['Unnamed: 0', 'id'], inplace=True)

# === Step 3: Show columns and data sample to help pick target ===
print("Columns in dataset:", df.columns.tolist())
print("\nSample data:\n", df.head())

# === Step 4: Set your target column ===
target_column = 'satisfaction'  # from your dataset info

if target_column not in df.columns:
    raise Exception(f"Target column '{target_column}' NOT found in dataset columns!")

# === Step 5: Handle missing values ===
df.dropna(inplace=True)

# === Step 6: Identify categorical columns except target ===
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
if target_column in cat_cols:
    cat_cols.remove(target_column)

# === Step 7: Encode categorical columns ===
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Encode target if categorical
if df[target_column].dtype == 'object':
    df[target_column] = le.fit_transform(df[target_column])

# === Step 8: Prepare features and target ===
X = df.drop(columns=[target_column])
y = df[target_column]

# === Step 9: Split data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 10: Train multiple models and evaluate ===
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
    'NaiveBayes': GaussianNB()
}

best_model = None
best_accuracy = 0

print("\n=== Model Performance ===")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# === Step 11: Save the best model ===
os.makedirs('model', exist_ok=True)
joblib.dump(best_model, 'model/booking_model.pkl')
print(f"\n✅ Best model '{best_model.__class__.__name__}' saved at 'model/booking_model.pkl'")
