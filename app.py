from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

pipeline = joblib.load('xgboostmodel_tuned.pkl')
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



@app.route("/chart-data")
def chart_data():
    df = pd.read_csv("car.csv")
    data = df[["km_driven", "selling_price"]].sample(500,random_state=42).to_dict(orient="records")

    return jsonify(data)
if __name__ == '__main__':
    app.run()
