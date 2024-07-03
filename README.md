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
    ```bash
    git clone https://github.com/yourusername/course_enrollment_prediction.git
    cd course_enrollment_prediction
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Flask application:**
    ```bash
    python app.py
    ```

5. **Access the application:**
    Open your web browser and go to `http://127.0.0.1:5000`.

## Project Structure

course_enrollment_prediction/
├── static/
│ └── styles.css
├── templates/
│ └── index.html
├── data/
│ └── cleaned_merged_data.csv
├── app.py
├── make_predictions.py
├── model_training.py
├── requirements.txt
└── README.md


## Usage

1. **Navigate to the Home page:**
    - Fill out the form with course data such as course code, term code, delivery code, capacity, year, semester, and category.
    - Hover over the information icons for tips on each field.

2. **Submit the form:**
    - The application will predict the enrollment for the specified course and display the predicted enrollment alongside a visualization comparing historical and predicted enrollments.

## Handling Unseen Labels

The model and application can handle previously unseen labels in the input data dynamically, ensuring that predictions are not limited by known labels.

## About Hudson University
Hudson University is a fictional university used in this model as a prototype.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



