from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load model and preprocessor
with open("best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocesser = pickle.load(f)

# Home route → load form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        Year = int(request.form['Year'])
        average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
        pesticides_tonnes = float(request.form['pesticides_tonnes'])
        avg_temp = float(request.form['avg_temp'])
        Area = request.form['Area']
        Item = request.form['Item']

        # Prepare features
        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)

        # Transform features
        transformed_features = preprocesser.transform(features)

        # Predict
        predicted_yield = best_model.predict(transformed_features)[0]

        # Convert to tons/ha
        tons_per_ha = (predicted_yield * 0.1) / 1000

        return render_template('index.html', 
                               prediction_text=f"Predicted Yield: {predicted_yield:.2f} hg/ha (~{tons_per_ha:.4f} tons/ha)")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
