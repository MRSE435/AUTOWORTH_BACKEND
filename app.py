from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

pipeline = joblib.load('linearmodel.pkl')
@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])

    log_price = pipeline.predict(input_df)[0]
    price = np.expm1(log_price)

    return jsonify({'predicted_price': round(float(price), 2)})

if __name__ == '__main__':
    app.run()
