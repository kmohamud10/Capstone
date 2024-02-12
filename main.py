from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load data
data = pd.read_csv('data/Cab_data.csv')

# Preprocess data
X = data[['KM_Travelled']]  # Select only the KM_Travelled column as input
y = data['Cost of Trip']

# Train the model
model = LinearRegression()
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_result():
  # Get input data from the form
  km_travelled = float(request.form['km_travelled'])

  # Predict using the model
  prediction = model.predict([[km_travelled]])[0]
  formatted_prediction = "{:.2f}".format(prediction)

  return render_template('index.html', prediction=formatted_prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
