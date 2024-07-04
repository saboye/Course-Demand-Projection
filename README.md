<div align="center">
  <h1>Course Enrollment Prediction</h1>
</div>

<p align="center">
    <img src="https://img.shields.io/github/contributors/saboye/Course-Demand-Projection?color=blue&logo=github&style=for-the-badge" alt="GitHub contributors" />
    <img src="https://img.shields.io/github/forks/saboye/Course-Demand-Projection?logo=github&style=for-the-badge" alt="GitHub forks" />
    <img src="https://img.shields.io/github/issues-raw/saboye/Course-Demand-Projection?style=for-the-badge" alt="GitHub issues" />
    <img src="https://img.shields.io/github/license/saboye/Course-Demand-Projection?style=for-the-badge" alt="GitHub license" />
    <img src="https://img.shields.io/github/last-commit/saboye/Course-Demand-Projection?style=for-the-badge" alt="GitHub last commit" />
    <img src="https://img.shields.io/badge/flask-1.1.2-blue?style=for-the-badge&logo=flask" alt="Flask" />
    <img src="https://img.shields.io/badge/scikit--learn-0.24.2-blue?style=for-the-badge&logo=scikit-learn" alt="scikit-learn" />
    <img src="https://img.shields.io/badge/pandas-1.2.4-blue?style=for-the-badge&logo=pandas" alt="Pandas" />
    <img src="https://img.shields.io/badge/numpy-1.20.3-blue?style=for-the-badge&logo=numpy" alt="NumPy" />
</p>

<p align="center">
  A machine learning project to forecast course enrollments using historical data. Features data cleaning, exploratory data analysis, model development with RandomForestRegressor, and deployment via a Flask web application.
</p>


# Course Enrollment Prediction

## Overview

This project aims to forecast course enrollments for collage or  University using machine learning. The predictive model uses historical enrollment data and various features such as course code, term code, delivery method, and capacity to forecast future enrollments. This allows the university to better allocate resources and ensure that courses are adequately staffed and resourced.

## Features

- Predict future course enrollments based on historical data.
- Visualize historical vs predicted enrollments.
- Handle unseen labels dynamically.
- User-friendly web interface using Flask.
- Tooltip hints for form inputs to guide users.

## Installation

1. **Clone the repository:**
    ```ruby
    git clone https://github.com/saboye/Course-Demand-Projection.git
    cd Course-Demand-Projection
    ```

2. **Create and activate a virtual environment:**
    ```ruby
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**
    ```ruby
    pip install -r requirements.txt
    ```

4. ### Model
Due to large file sizes, the model files are stored externally. Please download the following files and place them in the specified directories:

1. **Model File**: `best_random_forest_model.zip`
   - [Download Link]([https://your-storage-service.com/best_random_forest_model.zip](https://github.com/saboye/Course-Demand-Projection/blob/main/best_random_forest_model.zip))
   - Place the file in the root directory of the project and unzip it.

### Usage
1. Download the necessary files as mentioned above.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

4. **Run the Flask application:**
    ```ruby
    python app.py
    ```

5. **Access the application:**
    Open your web browser and go to `http://127.0.0.1:5000`.

## Project Structure
```ruby

course_enrollment_prediction/
├── data/                               # Directory for data files
├── EDA/                                # Directory for Exploratory Data Analysis
│   ├── cleaned_merged_data.csv         # Cleaned and merged data file
│   ├── Data Cleaning and Exploratory Data analysis (EDA).ipynb  # EDA notebook
├── static/                             # Directory for static files (CSS, JS, images)
│   ├── styles.css                      # CSS file for styling
├── templates/                          # Directory for HTML templates
│   ├── index.html                      # Main HTML file for the web application
├── venv/                               # Virtual environment directory
├── .gitignore                          # Git ignore file
├── app.py                              # Main Flask application
├── best_random_forest_model.joblib     # Trained RandomForest model
├── data_preprocessing.py               # Script for data preprocessing
├── label_encoders.joblib               # Label encoders used for encoding categorical data
├── make_predictions.py                 # Helper functions for encoding and predicting
├── model_training.py                   # Script for training the model
├── README.md                           # Project README file
├── requirements.txt                    # Python dependencies

```

## Usage

1. **Navigate to the Home page:**
    - Fill out the form with course data such as course code, term code, delivery code, capacity, year, semester, and category.
    - Hover over the information icons for tips on each field.

2. **Submit the form:**
    - The application will predict the enrollment for the specified course and display the predicted enrollment alongside a visualization comparing historical and predicted enrollments.

## Handling Unseen Labels

The model and application can handle previously unseen labels in the input data dynamically, ensuring that predictions are not limited by known labels.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



