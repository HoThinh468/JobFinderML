from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json

app = Flask(__name__)

# prediction function
def ValuePredictor(query):
    loaded_model = load(open("./model/model.pkl", "rb"))
    result = loaded_model.predict(query)
    return result[0]

@app.route("/result", methods=['POST'])
def predict():
    if request.method == 'POST':
        content = request.get_json(force = True)
        df = pd.json_normalize(json.loads(content))
        le = LabelEncoder()
        df['title'] = le.fit_transform(df['title'])
        df['location'] = le.fit_transform(df['location'])
        df['company_profile'] = le.fit_transform(df['company_profile'])
        df['requirements'] = le.fit_transform(df['requirements'])
        df['employment_type'] = le.fit_transform(df['employment_type'])
        df['required_experience'] = le.fit_transform(df['required_experience'])
        df['required_education'] = le.fit_transform(df['required_education'])
        df['industry'] = le.fit_transform(df['industry'])
        df['function'] = le.fit_transform(df['function'])
        df['description'] = le.fit_transform(df['description'])
        is_fraud = ValuePredictor(df)
        return jsonify(predict = str(is_fraud))
    else:
        return 'None'

@app.route("/", methods=['POST'])
def welcome():
        return "Welcome to job finder"

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)