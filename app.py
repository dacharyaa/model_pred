import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso,RidgeCV,LarsCV,ElasticNet,ElasticNetCV,LinearRegression,LassoCV
import statsmodels.api as sm
from  Model_Logic import mod_log

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST']) # To render Homepage
def home_page():
    return render_template('index.html')


@app.route('/insert', methods=['GET','POST'])  # insert a player to DB
def insert():
    print('AirTemp Prediction---------')

    if (request.method=='POST'):
        p1 = float(request.form['process'])
        p2 = float(request.form['rotational'])
        p3 = float(request.form['torque'])
        p4 = float(request.form['tool'])
        p5 = float(request.form['twf'])
        p6 = float(request.form['hdf'])
        p7 = float(request.form['pwf'])
        p8 = float(request.form['osf'])
        p9 = float(request.form['rnf'])
        print('process -->>>'+ str(p1))
        print('rotational speed -->>>'+  str(p2))
        res = mod_log(p1,p2,p3,p4,p5,p6,p7,p8,p9)
        print(res)
        return render_template('result.html', res= res)

if __name__ == '__main__':
    app.run(debug=True)
