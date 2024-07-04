import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Function to load data from a CSV file
def load_data(file_path):
    """
    Loads data from a CSV file.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    DataFrame: The loaded data as a pandas DataFrame.
    """
    return pd.read_csv(file_path)

# Function to preprocess data by removing rows with missing target values
def preprocess_data(data, target):
    """
    Preprocesses the data by dropping rows with missing values in the target column.
    
    Parameters:
    data (DataFrame): The input data as a pandas DataFrame.
    target (str): The name of the target column.
    
    Returns:
    DataFrame: The preprocessed data with rows containing missing target values removed.
    """
    data = data.dropna(subset=[target])
    return data

# Function to encode categorical features using LabelEncoder
def encode_features(data, features):
    """
    Encodes categorical features in the data using LabelEncoder.
    
    Parameters:
    data (DataFrame): The input data as a pandas DataFrame.
    features (list): A list of feature names to be encoded.
    
    Returns:
    DataFrame: The data with encoded features.
    dict: A dictionary of LabelEncoders for each encoded feature.
    """
    label_encoders = {}
    for feature in features:
        if data[feature].dtype == 'object':
            le = LabelEncoder()
            data[feature] = le.fit_transform(data[feature])
            label_encoders[feature] = le
    return data, label_encoders

# Function to split data into training and testing sets
def split_data(data, features, target, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.
    
    Parameters:
    data (DataFrame): The input data as a pandas DataFrame.
    features (list): A list of feature names to be used as predictors.
    target (str): The name of the target column.
    test_size (float): The proportion of data to be used as the test set.
    random_state (int): Random seed for reproducibility.
    
    Returns:
    tuple: Training and testing sets for features and target.
    """
    X = data[features]
    y = data[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Function to train a RandomForestRegressor model
def train_model(X_train, y_train, params):
    """
    Trains a RandomForestRegressor model with given parameters.
    
    Parameters:
    X_train (DataFrame): The training feature data.
    y_train (Series): The training target data.
    params (dict): Parameters for the RandomForestRegressor.
    
    Returns:
    RandomForestRegressor: The trained model.
    """
    model = RandomForestRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    return model

# Function to evaluate a trained model on test data
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on test data.
    
    Parameters:
    model (RandomForestRegressor): The trained model.
    X_test (DataFrame): The test feature data.
    y_test (Series): The test target data.
    
    Returns:
    tuple: Mean Squared Error, Mean Absolute Error, and R² Score of the model predictions.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2

# Function to find the best hyperparameters for the model
def find_best_params(X_train, y_train, X_test, y_test, param_grid):
    """
    Finds the best parameters for the RandomForestRegressor model using a parameter grid.
    
    Parameters:
    X_train (DataFrame): The training feature data.
    y_train (Series): The training target data.
    X_test (DataFrame): The test feature data.
    y_test (Series): The test target data.
    param_grid (list): A list of parameter dictionaries to try.
    
    Returns:
    tuple: Best parameters, best MSE, and the best model.
    """
    best_mse = float('inf')
    best_params = None
    best_model = None
    for params in param_grid:
        model = train_model(X_train, y_train, params)
        mse, mae, r2 = evaluate_model(model, X_test, y_test)
        if mse < best_mse:
            best_mse = mse
            best_params = params
            best_model = model
    return best_params, best_mse, best_model

# Function to save the trained model to disk
def save_model(model, filename):
    """
    Saves the trained model to a file.
    
    Parameters:
    model (RandomForestRegressor): The trained model.
    filename (str): The file path to save the model.
    """
    joblib.dump(model, filename)

# Function to save the label encoders to disk
def save_label_encoders(label_encoders, filename):
    """
    Saves the label encoders to a file.
    
    Parameters:
    label_encoders (dict): A dictionary of LabelEncoders.
    filename (str): The file path to save the label encoders.
    """
    joblib.dump(label_encoders, filename)

# Function to encode new data
def encode_new_data(new_data, features, label_encoders):
    """
    Encodes new data using the provided label encoders.
    
    Parameters:
    new_data (DataFrame): The new data to be encoded.
    features (list): A list of feature names to be encoded.
    label_encoders (dict): A dictionary of LabelEncoders.
    
    Returns:
    DataFrame: The encoded new data.
    """
    for feature in features:
        if new_data[feature].dtype == 'object':
            le = label_encoders[feature]
            new_data[feature] = le.transform(new_data[feature])
    return new_data

# Function to make predictions on new data
def make_predictions(model, new_data, features):
    """
    Makes predictions on new data using the trained model.
    
    Parameters:
    model (RandomForestRegressor): The trained model.
    new_data (DataFrame): The new data to predict on.
    features (list): A list of feature names to be used for prediction.
    
    Returns:
    DataFrame: The new data with an additional column for predicted enrollments.
    """
    new_X = new_data[features]
    predictions = model.predict(new_X)
    new_data['predicted_enrollment'] = predictions
    return new_data

# Usage example
if __name__ == "__main__":
    # Load data
    file_path = 'data/cleaned_merged_data.csv'  
    data = load_data(file_path)

    # Define target and features
    target = 'enrolled'
    features = ['course_code', 'term_code', 'delivery_code', 'cap_x', 'year', 'semester', 'category']

    # Preprocess data
    data = preprocess_data(data, target)

    # Encode features
    data, label_encoders = encode_features(data, features)

    # Split data
    X_train, X_test, y_train, y_test = split_data(data, features, target)

    # Define parameter grid
    param_grid = [
        {'n_estimators': 100, 'max_depth': None},
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 200, 'max_depth': 10},
        {'n_estimators': 200, 'max_depth': None},
    ]

    # Find best parameters and train model
    best_params, best_mse, best_model = find_best_params(X_train, y_train, X_test, y_test, param_grid)

    # Evaluate the best model
    mse, mae, r2 = evaluate_model(best_model, X_test, y_test)
    print(f'Best Params: {best_params}')
    print(f'MSE: {mse}, MAE: {mae}, R2: {r2}')

    # Save the model
    model_filename = 'best_random_forest_model.joblib'
    save_model(best_model, model_filename)

    # Save the label encoders
    label_encoders_filename = 'label_encoders.joblib'
    save_label_encoders(label_encoders, label_encoders_filename)

    # Example new data for predictions
    new_course_data = pd.DataFrame({
        'course_code': ['BIOL 181', 'ALST 202'],
        'term_code': [202201, 202201],
        'delivery_code': ['DCAM', 'DCAM'],
        'cap_x': [80, 16],
        'year': [2022, 2022],
        'semester': ['Spring', 'Spring'],
        'category': ['General', 'General']
    })

    # Encode new data
    new_course_data = encode_new_data(new_course_data, features, label_encoders)

    # Make predictions
    predictions = make_predictions(best_model, new_course_data, features)
    print(predictions)
