import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

data = pd.read_csv('https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/DAAG/possum.csv').iloc[:, 1:]
# print(data)
myLabel = LabelEncoder()
data['Pop'] = myLabel.fit_transform(data['Pop'])
data['sex'] = myLabel.fit_transform(data['sex'])

data.dropna(axis=0, inplace=True)
# print(data.head())
# print(data.isnull().sum)

y = data['footlgth'].values
X = data.drop('footlgth', axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
hybrid = Pipeline([("scale", StandardScaler()),
                   ("PCA", PCA()),
                    ("Algorithm", LinearRegression())])
hybrid.fit(X_train, y_train)
print(hybrid.score(X_test, y_test))
print(np.sum(hybrid.named_steps['PCA'].explained_variance_ratio_))
# me9 leqciaze garcheuli kodi. mzadeba me3 quizistvis. Novemeber 17. es kodi meilze maq
