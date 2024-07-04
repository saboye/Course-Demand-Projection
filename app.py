from flask import Flask, request, render_template
import pandas as pd
import joblib
import plotly.express as px
import plotly.io as pio
from make_predictions import encode_new_data

app = Flask(__name__)

# Load the trained model
model_filename = 'best_random_forest_model.joblib'
loaded_model = joblib.load(model_filename)

# Load label encoders
label_encoders_filename = 'label_encoders.joblib'
label_encoders = joblib.load(label_encoders_filename)

# Define features
features = ['course_code', 'term_code', 'delivery_code', 'cap_x', 'year', 'semester', 'category']

@app.route('/')
def home():
    """
    Renders the home page with the input form.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles form submission, predicts enrollment, and displays the result with a visualization.
    """
    try:
        # Get data from the form
        data = {
            'course_code': [request.form['course_code']],
            'term_code': [int(request.form['term_code'])],
            'delivery_code': [request.form['delivery_code']],
            'cap_x': [int(request.form['cap_x'])],
            'year': [int(request.form['year'])],
            'semester': [request.form['semester']],
            'category': [request.form['category']]
        }
        
        new_course_data = pd.DataFrame(data)
        course_code = request.form['course_code']
        year = request.form['year']
        term_code_str = f"{year}01"  # '01' as a placeholder for the Semester

        # Encode the new data
        new_course_data = encode_new_data(new_course_data, features, label_encoders)

        # Make predictions
        predictions = loaded_model.predict(new_course_data[features])
        rounded_prediction = round(predictions[0])

        # Load historical data for visualization
        historical_data = pd.read_csv('data/cleaned_merged_data.csv')
        historical_data_filtered = historical_data[
            (historical_data['course_code'] == data['course_code'][0])
        ]

        historical_data_filtered['type'] = 'Historical'
        historical_data_filtered['term_code'] = pd.to_datetime(historical_data_filtered['term_code'].astype(str), format='%Y%m')

        # Convert new course term_code for proper alignment
        new_course_data['term_code'] = pd.to_datetime(term_code_str, format='%Y%m')
        new_course_data['enrolled'] = rounded_prediction
        new_course_data['type'] = 'Predicted'

        # Concatenate the historical data with the new course data
        combined_data = pd.concat([historical_data_filtered, new_course_data], ignore_index=True)

        # Create the plot
        fig = px.line(
            combined_data,
            x='term_code',
            y='enrolled',
            color='type',
            title=f'Enrollment Trends for {course_code}',
            labels={'term_code': 'Term', 'enrolled': 'Enrollment'}
        )

        # Ensure that markers are used to indicate points on the plot
        fig.update_traces(mode='markers+lines')
        
        fig.update_layout(
            xaxis_title='Term',
            yaxis_title='Enrollment',
            legend_title='Data Type',
            title={'x': 0.5, 'xanchor': 'center'}
        )

        # Ensure the x-axis range includes the year 2024
        fig.update_xaxes(range=[pd.to_datetime('2017-01-01'), pd.to_datetime('2024-12-31')])

        # Convert plot to HTML
        graph_html = pio.to_html(fig, full_html=False)

        return render_template(
            'index.html',
            prediction=f'{course_code} Predicted Enrollment for {year} is: {rounded_prediction}',
            graph_html=graph_html
        )
    except Exception as e:
        return render_template('index.html', error=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
