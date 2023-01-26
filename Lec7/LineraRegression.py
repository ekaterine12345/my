import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('https://raw.githubusercontent.com/alihussainmeer/Data-Analysis-to-predict-the-CO2-Emission/master/FuelConsumption.csv')
# print(data)
# print(data.isnull().any())
data.drop(["MAKE", "MODEL", "VEHICLECLASS", "MODELYEAR"], axis=1, inplace=True)
print(data.isnull().any())

from sklearn.preprocessing import LabelEncoder
myLabel = LabelEncoder()
data["TRANSMISSION"] = myLabel.fit_transform(data["TRANSMISSION"])
data["FUELTYPE"] = myLabel.fit_transform(data["FUELTYPE"]) # transform-მა არ იმუშავა
y = data["CO2EMISSIONS"].values
X = data[["CYLINDERS", "TRANSMISSION", "FUELTYPE", "FUELCONSUMPTION_CITY"]].values
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
myLinearModel = LinearRegression()
myLinearModel.fit(X_train, y_train)
print(myLinearModel.score(X_test, y_test))
# me7 leqciaze dawerili kodi