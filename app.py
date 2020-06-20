

from flask import Flask, render_template, request,send_file

import pandas as pd

app = Flask(__name__)

import pickle
test_preprocessed = pd.read_csv('test_preprocessed.csv')
test_preprocessed = test_preprocessed.drop(['Unnamed: 0'],axis =1)
test_preprocessed = test_preprocessed.iloc[:,:].values

@app.route('/')
def index():
 
    return render_template('index.html')

@app.route('/regressor', methods = ['POST','GET'])
def regressor():
    if request.method == 'POST':
        if 'DT' in request.form:
 
            regressor_dt = pickle.load(open('regressorDT.sav','rb'))
            y_pred1 = regressor_dt.predict(test_preprocessed)
            output = pd.DataFrame(y_pred1)
            output.to_csv('decision_tree.csv')
            path = "dt.csv"
                            
            return send_file(path,as_attachment = True)
        
        if 'reg' in request.form:
            regressor = pickle.load(open('regressor.sav','rb'))
            y_pred1 = regressor.predict(test_preprocessed)
            output = pd.DataFrame(y_pred1)
            output.to_csv('regression.csv')
            path = "regression.csv"
            return send_file(path ,as_attachment = True)
        
        if 'Rf' in request.form:
            regressor_rf = pickle.load(open('regressorRF.sav','rb'))
            y_pred1 = regressor_rf.predict(test_preprocessed)
            output = pd.DataFrame(y_pred1)
            output.to_csv('random_forest.csv')
            path = "random_forest.csv"
            return send_file(path ,as_attachment = True)
        
        if 'Ada' in request.form:
            adaboost = pickle.load(open('regressorRF_ada.sav','rb'))
            y_pred1 = adaboost.predict(test_preprocessed)
            output = pd.DataFrame(y_pred1)
            output.to_csv('adaboost.csv')
            path = "adaboost.csv"
            return send_file(path ,as_attachment = True)
        
        if 'Gb' in request.form:
            gb = pickle.load(open('est.sav','rb'))
            y_pred1 = gb.predict(test_preprocessed)
            output = pd.DataFrame(y_pred1)
            output.to_csv('gradient_boost.csv')
            path = "gradient_boost.csv"
            return send_file(path ,as_attachment = True)
            
    
    
    return render_template("index.html")

if __name__ == "__main__":
    
    app.run(debug=True)
    
    
    
    
    
    
    
    
    
    
    
    
    