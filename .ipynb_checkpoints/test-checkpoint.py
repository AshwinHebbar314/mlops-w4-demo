import pytest
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Import the function to be tested
from train import train_model, DATA_CSV_PATH, LOCAL_MODEL_OUTPUT_PATH

# --- Fixtures for setting up test conditions ---

@pytest.fixture
def raw_data():
    """Fixture to load the raw iris dataset."""
    assert os.path.exists(DATA_CSV_PATH), "Iris dataset not found at specified path."
    data = pd.read_csv(DATA_CSV_PATH)
    return data

# --- Data Validation Tests ---

def test_data_loading_and_shape(raw_data):
    """Tests if the data loads correctly and has the expected number of rows and columns."""
    assert isinstance(raw_data, pd.DataFrame), "Data is not a pandas DataFrame."
    # Iris dataset has 150 samples and 5 columns (4 features + 1 target)
    assert raw_data.shape[0] >= 150, "Dataset should have at least 150 rows."
    assert raw_data.shape[1] == 5, "Dataset should have 5 columns."

def test_data_columns(raw_data):
    """Tests for the presence and correctness of expected columns."""
    expected_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    assert all(col in raw_data.columns for col in expected_columns), "Dataset is missing one or more required columns."

def test_no_null_values(raw_data):
    """Tests to ensure there are no missing values in the dataset."""
    assert raw_data.isnull().sum().sum() == 0, "There are null values in the dataset."

def test_target_column_distribution(raw_data):
    """Checks if the target column 'species' has the three expected classes."""
    expected_species = {'setosa', 'versicolor', 'virginica'}
    assert set(raw_data['species'].unique()) == expected_species, "Target column 'species' has unexpected values."


# --- Model Training and Evaluation Sanity Tests ---

def test_train_model_script_runs_successfully():
    """
    Tests if the main training function runs without raising any exceptions
    and creates the model artifact.
    """
    # Clean up previous artifacts if they exist
    if os.path.exists(LOCAL_MODEL_OUTPUT_PATH):
        os.remove(LOCAL_MODEL_OUTPUT_PATH)
    
    # Run the training function
    try:
        train_model()
    except Exception as e:
        pytest.fail(f"train_model() raised an exception: {e}")

    # Check if the model artifact was created
    assert os.path.exists(LOCAL_MODEL_OUTPUT_PATH), "Model artifact was not created after training."

def test_model_performance_sanity_check():
    """
    This is a sanity test to ensure the trained model's accuracy is above a
    reasonable threshold. For the Iris dataset, a simple model should be quite accurate.
    """
    # Ensure the model is trained and available
    if not os.path.exists(LOCAL_MODEL_OUTPUT_PATH):
        train_model()

    # Load the trained model
    model = joblib.load(LOCAL_MODEL_OUTPUT_PATH)
    assert isinstance(model, DecisionTreeClassifier), "Loaded artifact is not a DecisionTreeClassifier."

    # Load the data and create the same test split as in the training script
    data = pd.read_csv(DATA_CSV_PATH)
    X_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    y_target = 'species'
    
    _, test_df = train_test_split(data, test_size=0.2, stratify=data[y_target], random_state=42)
    X_test = test_df[X_features]
    y_test = test_df[y_target]
    
    # Make predictions and check accuracy
    predictions = model.predict(X_test)
    accuracy = (predictions == y_test).mean()

    # Sanity check: Assert that accuracy is reasonably high (e.g., > 90%)
    assert accuracy > 0.70, f"Model accuracy {accuracy:.2f} is below the 50% threshold."