from flask import Flask, request, jsonify
import json
import pandas as pd
from joblib import load
from CreditCardFraudModel.preprocess import preprocessing
from CreditCardFraudModel.model import RunModel
import os

from sklearn.base import TransformerMixin
            
app = Flask(__name__)  

@app.route("/", methods=['GET'])
def index():
    # RunModel()
    return "<h1>Welcome to the credit card fraud detection API</h1>"

@app.route("/check/", methods=['GET'])
def checkfraud():
    data = request.args.get('data', default=None, type=str)
    data = " ".join(data.split()).split()
    columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
    dic = dict()
    for value,col in zip(data,columns):
        dic[col] = [value]
    print(data)
    dataframe = pd.DataFrame(dic)
    file_path = os.path.join(os.path.dirname(__file__), 'creditcardfraudmodel.pkl')
    fraud_model = load(file_path)

    prediction = fraud_model.predict(dataframe)

    result_dic = {"transaction": 'fraud' if prediction[0] == 1 else 'not fraud'}
    
    return jsonify(result_dic)