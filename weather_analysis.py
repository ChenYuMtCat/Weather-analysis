import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('weatherHistory.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 0] = le.fit_transform(X[:, 0])

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#Evaluating the Model Performance
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

#Visualising the Training set results
plt.scatter(X_train[:, 0], y_train, color = 'red')
plt.plot(X_train[:, 0], regressor.predict(X_train), color = 'blue')
plt.title('Temperature vs Humidity (Training set)')
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.show()

#Visualising the Test set results
plt.scatter(X_test[:, 0], y_test, color = 'red')
plt.plot(X_train[:, 0], regressor.predict(X_train), color = 'blue')
plt.title('Temperature vs Humidity (Test set)')
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.show()
