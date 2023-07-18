from flask import Flask, request, jsonify
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer

model = pickle.load(open('model1.pkl', 'rb'))
cv = pickle.load(open('CountVectorizer.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return "Hello world"


@app.route('/predict', methods=['POST'])
def predict():

    symptom = request.form.get('symptom')

    input_query = cv.transform([symptom]).toarray()

    result = model.predict(input_query)

    return jsonify({'Disease': result[0]})


if __name__ == '__main__':
    app.run(debug=True)
