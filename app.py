from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model from the pickle file
with open('energy_cost_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    energy_usage = float(request.form['energy_usage'])
    hvac_usage = float(request.form['hvac_usage'])
    lighting_usage = float(request.form['lighting_usage'])
    equipment_usage = float(request.form['equipment_usage'])
    
    # Create a feature array for the model
    features = np.array([[energy_usage, hvac_usage, lighting_usage, equipment_usage]])

    # Make a prediction using the model
    prediction = model.predict(features)

    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
