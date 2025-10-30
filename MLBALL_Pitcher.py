import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# LOAD TRAINING DATA
train_data = pd.read_csv('data_train_midterm_problem6.csv')
X_train = train_data[['x1', 'x2']]
y_train = train_data['y']

# LOAD TEST DATA
test_data = pd.read_csv('data_test_midterm_problem6.csv')
X_test = test_data[['x1', 'x2']]

# USE POLYNOMIAL FEATURES TO CAPTURE NONLINEARITY
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# TRAIN LINEAR REGRESSION MODEL
model = LinearRegression()
model.fit(X_train_poly, y_train)

# MAKE PREDICTIONS
predictions = model.predict(X_test_poly)

# SAVE PREDICTIONS TO CSV
pd.DataFrame(predictions).to_csv('gutierrez_predictions_midterm_problem6.csv', index=False, header=False)