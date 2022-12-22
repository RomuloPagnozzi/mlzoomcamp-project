import pickle
import xgboost as xgb
from flask import Flask
from flask import request
from flask import jsonify

model_file = 'xgb_13aucpr.pkl'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('fraud')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    features = dv.get_feature_names_out()
    X = dv.transform(customer)
    dX = xgb.DMatrix(X, feature_names=features)
    y_pred = model.predict(dX)[0].round(3)
    fraud = y_pred >= 0.5
    
    result = {
        'fraud_probability': float(y_pred),
        'fraud': bool(fraud)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
