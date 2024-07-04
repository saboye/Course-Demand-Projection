import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Function to encode new data, dynamically adding unseen classes to the label encoder
def encode_new_data(new_data, features, label_encoders):
    """
    Encodes new data using the provided label encoders. Dynamically adds new classes to the encoders if they are not seen before.
    
    Parameters:
    new_data (DataFrame): The new data to encode.
    features (list): A list of feature names to encode.
    label_encoders (dict): A dictionary of LabelEncoders for each feature.
    
    Returns:
    DataFrame: The encoded new data.
    """
    for feature in features:
        if new_data[feature].dtype == 'object':
            le = label_encoders[feature]
            known_classes = set(le.classes_)
            new_classes = set(new_data[feature]) - known_classes

            if new_classes:
                # Add new classes to the encoder
                le.classes_ = np.append(le.classes_, list(new_classes))

            # Apply the label encoder to the new data
            new_data[feature] = new_data[feature].apply(lambda x: le.transform([x])[0])
    return new_data

# Function to make predictions on new data using the trained model
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

