import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
from flask import Flask, request, render_template

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    try:
        input_features = [float(x) for x in request.form.values()]
        feature_values = [np.array(input_features)] 
        names = [['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month','day','hours', 'minutes', 'seconds']]
        data = pd.DataFrame(feature_values, columns=names)
        data = scaler.fit_transform(data) 
        data = pd.DataFrame(data, columns=names) 
        prediction = model.predict(data)
        
        result = f"Estimated Traffic Volume is: {int(prediction[0])}"
        return render_template('result.html', prediction_text=result)
        
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")
        

if __name__ == '__main__':
    port = int(os.environ.get('PORT',5000))
    app.run(port=port, debug = True, use_reloader = False)

