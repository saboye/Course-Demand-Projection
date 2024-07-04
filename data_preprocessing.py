import pandas as pd
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
