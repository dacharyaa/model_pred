import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso,RidgeCV,LarsCV,ElasticNet,ElasticNetCV,LinearRegression,LassoCV
import statsmodels.api as sm
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def mod_log(p1,p2,p3,p4,p5,p6,p7,p8,p9):
    try:

            df11 = pd.read_csv('ai4i2020.csv')
            print(df11.head())
            Y = df11["Air temperature [K]"]
            X = df11.drop(columns=['UDI', 'Product ID', 'Type', 'Air temperature [K]', 'Machine failure'])
            # mu =0 and stndarsd deviation=1
            scaler = StandardScaler()
            arr = scaler.fit_transform(X)
            #VIF
            vif_df = pd.DataFrame()
            vif_df['vif'] = [variance_inflation_factor(arr, i) for i in range(arr.shape[1])]
            vif_df['feature'] = X.columns
            print(vif_df)
            #split to train and test dataset
            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test = train_test_split(arr, Y, test_size=0.25, random_state=100)
            # create the model
            lr = LinearRegression()
            lr.fit(x_train, y_train)
            # check the score
            print(lr.score(x_train, y_train))
            #test data
            print(lr.predict(scaler.transform([[308.6, 1551, 42.8, 0, 0, 0, 0, 0, 0]])))
            # test user entered data
            data = lr.predict(scaler.transform([[p1,p2,p3,p4,p5,p6,p7,p8,p9]]))
            return data
    except Exception as e:
           print('Exception as -->',e)