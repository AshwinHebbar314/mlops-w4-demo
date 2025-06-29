import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import os

DATA_CSV_PATH = 'data/iris.csv'
LOCAL_MODEL_OUTPUT_PATH = 'artifacts/model.joblib'

def train_model():
    data = pd.read_csv(DATA_CSV_PATH)
    print(f"Data loaded successfully. Shape: {data.shape}")

    os.makedirs(os.path.dirname(LOCAL_MODEL_OUTPUT_PATH), exist_ok=True)
    
    X_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    y_target = 'species'

    test_set_size = 0.2 

    train_df, test_df = train_test_split(data, test_size=test_set_size, stratify=data[y_target], random_state=42)
    print(f"Data split: Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    X_train = train_df[X_features]
    y_train = train_df[y_target]
    X_test = test_df[X_features]
    y_test = test_df[y_target]

    model = DecisionTreeClassifier(max_depth=3, random_state=1)
    model.fit(X_train, y_train)
    print(f"Model trained: {model}")

    prediction = model.predict(X_test)
    accuracy = metrics.accuracy_score(prediction, y_test)
    print(f"\nModel Accuracy: {accuracy:.4f}")

    report = metrics.classification_report(y_test, prediction, zero_division=0)
    print("\nClassification Report:")
    print(report)

    joblib.dump(model, LOCAL_MODEL_OUTPUT_PATH)
    print(f"Trained model saved to: {LOCAL_MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    print("--- Running Iris Model Training ---")
    train_model()