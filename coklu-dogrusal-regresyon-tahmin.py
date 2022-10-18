import pandas as pd
df = pd.read_csv("Advertising.csv")
df = df.iloc[:,1:len(df)]
df.head()

X = df.drop("sales", axis=1)
y = df[["sales"]]

# statsmodel ile model kurmak

import statsmodels.api as sm

lm = sm.OLS(y, X)
model = lm.fit()
print(model.summary())

# scikit learn ile model oluşturmak
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(X,y)

print(model.intercept_)
print(model.coef_)

#****************************************

yeni_veri = [[10],[20],[30]]

import pandas as pd
yeni_veri = pd.DataFrame(yeni_veri).T

print(yeni_veri)

print(model.predict(yeni_veri))

from sklearn.metrics import mean_squared_error
import numpy as np

MSE = mean_squared_error(y, model.predict(X))
RMSE = np.sqrt(MSE)

print(MSE)
print(RMSE)