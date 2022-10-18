import pandas as pd

df = pd.read_csv("Advertising.csv")

df = df.iloc[:,1:len(df)]

print(df.head())

import seaborn as sns

sns.jointplot(x = "TV", y = "sales", data=df, kind="reg")

from sklearn.linear_model import LinearRegression
X = df[["TV"]]   #bağımsız değişken
y = df[["sales"]]   #bağımlı değişken

reg = LinearRegression() #model nesnesi oluşturma

model = reg.fit(X,y)

model.intercept_ # sabit

model.coef_ #katsayı

print("model sabiti: ",model.intercept_)

print("model katsayısı: ",model.coef_)

model.score(X,y)    #model skoru rkare  [Bağımlı değişkendeki değişikliğin, bağımsız değişkenlerce açıklanma yüzdesidir.]

print("model skoru: ",model.score(X,y))

#**************************************

# import matplotlib.pyplot as plt
# g = sns.regplot(df[["TV"]], df[["sales"]], ci=None, scatter_kws = {"color":"r", "s":9})
# g.set_title("Model Denklemi: Sales = 7.03 + TV*0.05 ")
# g.set_xlabel("TV Harcamaları")
# g.set_ylabel("Satış Harcamaları")
# plt.xlim(-10,310)
# plt.ylim(bottom=0)

harcamalar = [[10],[30],[50]]

print(model.predict(harcamalar))

#***************************************

# MSE : Hata Kareler Ortalaması
# RMSE : Hata Kareler Ortalamasının Karekökü

print(y.head())
print(model.predict(X)[0:5])

gercek_y = y[:10]
tahmin_edilen_y = pd.DataFrame(model.predict(X)[:10])

hatalar = pd.concat([gercek_y, tahmin_edilen_y], axis = 1)
hatalar.columns = ["gerçek y", "tahmin edilen y"]

hatalar["hata"] = hatalar["gerçek y"] - hatalar["tahmin edilen y"]

hatalar["hata kareler"] = hatalar["hata"]**2

import numpy as np

hatalar["hata kareler ortalaması"] = np.mean(hatalar["hata kareler"])

print(hatalar)

