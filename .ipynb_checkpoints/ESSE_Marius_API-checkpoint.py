from flask import Flask, jsonify, request

import pickle
import pandas as pd
from zipfile import ZipFile

app = Flask(__name__)

z = ZipFile("data/default_risk_ma.zip")
data = pd.read_csv(z.open('default_risk_ma.csv'), index_col='SK_ID_CURR', encoding ='utf-8')

z = ZipFile("data/X_sample.zip")
sample = pd.read_csv(z.open('X_sample.csv'), index_col='SK_ID_CURR', encoding ='utf-8')
print(sample.index)
# charger le modole a partir d'un pickcle
pickle_in = open('model/LGBMClassifier.pkl', 'rb') 
clf = pickle.load(pickle_in)
 
@app.route("/predict", methods=['GET'])
def get_prediction():
    
    clientId = request.args.get('clientId')
	# charger les données du client correspondanta clientid
    print(clientId)
	
	# prediction en utilisant le model chargé
    X=sample.iloc[:, :-1]
    score = clf.predict_proba(X[X.index == int(clientId)])[:,1][0]
    print(score)
    return jsonify(defaut=score)
 
if __name__ == '__main__':
    app.run()