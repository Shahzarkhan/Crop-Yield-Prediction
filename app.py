from flask import Flask,request, render_template
import numpy as np
import pandas as pd
import pickle
import sklearn

#loading models
rf = pickle.load(open('rf.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))

#flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')
@app.route("/predict",methods=['POST'])
def predict():
    if request.method == 'POST':
        Crop_Year = request.form['Crop_Year']
        Annual_Rainfall = request.form['Annual_Rainfall']
        Pesticide = request.form['Pesticide']
       # avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Crop = request.form['Crop']

        features = np.array([[Crop_Year, Annual_Rainfall,Pesticide, Area, Crop]],dtype=object)
        transformed_features = preprocessor.transform(features)
        predicted_value = rf.predict(transformed_features).reshape(1,-1)

        return render_template('index.html',predicted_value=predicted_value)

#python main
if __name__=="__main__":
    app.run(debug=True)