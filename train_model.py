import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv("insurance.csv")

# Convert text columns into numbers
data = pd.get_dummies(
    data,
    columns=["sex", "smoker", "region"],
    drop_first=True
)

# Separate input and output
X = data.drop("charges", axis=1)
y = data["charges"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model and column names
model_data = {
    "model": model,
    "columns": X.columns.tolist()
}

joblib.dump(model_data, "model.pkl")

print("Model trained successfully!")
print("Model saved as model.pkl")