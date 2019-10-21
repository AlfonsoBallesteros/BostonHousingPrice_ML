from flask import Flask, jsonify, request
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    message = {'id':123, 'name':'Flask test'}
    #return 'Hello World'
    return jsonify(message)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # load the model from disk
    filename = './Models/housing_svc.model'
    svc = joblib.load(filename)
    
    # load the scaler from disk
    filename = './Models/scaler.scaler'
    scaler = joblib.load(filename)

    content = request.get_json(force=True)
    print(content)

    CRIM = content['CRIM']
    ZN = content['ZN'] 
    INDUS = content['INDUS']
    CHAS = content['CHAS'] 
    NOX = content['NOX']
    RM = content['RM']
    AGE = content['AGE'] 
    DIS = content['DIS'] 
    RAD = content['RAD'] 
    TAX = content['TAX'] 
    PTRATIO = content['PTRATIO']
    B = content['B']
    LSTAT = content['LSTAT']
 
    #Make a Prediction
    # We create a new (fake) person having the three most correated values high
    new_df = pd.DataFrame([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])
    # We scale those values like the others
    new_df_scaled = scaler.transform(new_df)
    
    # We predict the outcome
    prediction = svc.predict(new_df_scaled)
    
    # A value of "1" means that this person is likley to have type 2 diabetes
    price_prediction = float(prediction)
    price = format(price_prediction, '.3f')
    
    return jsonify({'precio':price})
    
if __name__ == "__main__":
    app.run()
