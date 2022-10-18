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


