from cv2 import randShuffle
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

#*******************************************

print(X.head())
print(y.head())

# sınama seti
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=99)

print(X_train.head())
print(y_train.head())
print(X_test.head())
print(y_test.head())

lm = LinearRegression()
model = lm.fit(X_train, y_train)

# eğitim hatası
print(np.sqrt(mean_squared_error(y_train, model.predict(X_train))))

# test hatası
print(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))

# k katlı cross validation
from sklearn.model_selection import cross_val_score
print(cross_val_score(model, X_train, y_train, cv=10, scoring="neg_mean_squared_error"))

# ortalama hata mse
print("ortalama mse hata: ",np.mean(-cross_val_score(model, X_train, y_train, cv=10, scoring="neg_mean_squared_error")))

# ortalama hata rmse
print("ortalama rmse hata: ",np.sqrt(np.mean(-cross_val_score(model, X_train, y_train, cv=10, scoring="neg_mean_squared_error"))))
