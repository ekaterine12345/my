import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
data.drop(["PassengerId", "Ticket", "Name", "Cabin", "Embarked"], axis=1, inplace=True)


print(data.head())
myLabel = LabelEncoder()
data["Sex"] = myLabel.fit_transform(data['Sex'])

# გამოტოვებული ელემენტების შევსება საშუალოთი
data['Age'] = data['Age'].fillna(data['Age'].mean())
print(data.head())

X = data.drop("Survived", axis=1).values
y = data['Survived'].values
# SelectKBest ის მედოდის გამოყენებით ამოარჩიეთ 3 საუკეთესო სვეტი და
# ამ 3 სვეტზე გამოვთვალოთ სატრენინგო და სატესტო სქორი
X_selected = SelectKBest(score_func=f_classif, k=3).fit_transform(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=1)

# MinMaxScaler-ის გამოყენებით დააფორმატირეთ სატრენინგო და სატესტო მონაცემები
my_MinMax = MinMaxScaler()
X_train = my_MinMax.fit_transform(X_train)
X_test = my_MinMax.fit_transform(X_test)

# გამოვიყენოთ DecisionTreeClassifier ალგორითმი ბიბლიოთეკიდან sklearn.tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(model.score(X_train, y_train))

# PCA მეშვეობით შეამცირეთ განზომიება 2 სვეტამდე და გამოთვალეთ ახალი სქორი
my_PCA = PCA(n_components=2)
X_train = my_PCA.fit_transform(X_train)
X_test = my_PCA.fit_transform(X_test)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))