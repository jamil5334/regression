import pandas as pd
import numpy as np

data=pd.read_csv('Real estate.csv')
data.read()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

X = data.drop('Y house price of unit area', axis=1)
y = data['Y house price of unit area']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model=RandomForestRegressor()
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)

print('Random Forest MSE:{rf_mse}')
print('Random Forest MAE:{rf_mae}')
print('Random Forest RMSE:{rf_rmse}')

dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_predictions)
dt_mae = mean_absolute_error(y_test, dt_predictions)
dt_rmse = np.sqrt(dt_mse)

print('Decision Tree MSE: {dt_mse}')
print('Decision Tree MAE: {dt_mae}')
print('Decision Tree RMSE: {dt_rmse}')