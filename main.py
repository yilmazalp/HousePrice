# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np 

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


#melbourne_file_path = '../input/melb_data.csv'
#melbourne_data = pd.read_csv(melbourne_file_path)
train_data = pd.read_csv('C:/Users/User/PycharmProjects/spyder/HousePrice/train.csv')

imputer_data = Imputer()

data_pipeline = make_pipeline(Imputer(), RandomForestRegressor())

train_y = train_data.SalePrice
#train_y = imputer_data.fit_transform(train_y)
#train_y = train_y.fillna(train_y.mean())

#y = melbourne_data.Price
#y = y.fillna(y.mean())

#melbourne_predictors = ['Rooms', 'Car', 'Landsize', 'Bedroom2', 'Bathroom', 'BuildingArea',
#                        'YearBuilt', 'Lattitude', 'Longtitude', 'Propertycount']

predictors = ['LotArea', 'YearBuilt', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
              'GarageArea', 'TotRmsAbvGrd', 'OverallQual', 'GarageCars', 'OverallCond',
              'MoSold', 'YrSold', 'WoodDeckSF', 'OpenPorchSF', 'Fireplaces', 'TotRmsAbvGrd',
              'BsmtFinSF1', 'BsmtFinSF2', 'LotFrontage', 'GrLivArea', 'BsmtUnfSF', 'FullBath',
              'HalfBath', 'GarageYrBlt', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']

train_X = train_data[predictors]
train_X = imputer_data.fit_transform(train_X)
#train_X = train_X.fillna(train_X.mean())

#X = melbourne_data[melbourne_predictors]
#X = X.fillna(X.mean())

#train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 0)

test_data = pd.read_csv('C:/Users/User/PycharmProjects/spyder/HousePrice/test.csv')
test_X = test_data[predictors]
test_X = imputer_data.transform(test_X)
#test_X = test_X.fillna(test_X.mean())

forest_model = RandomForestRegressor()
#forest_model.fit(train_X, train_y)
data_pipeline.fit(train_X, train_y)

scores = cross_val_score(data_pipeline, train_X, train_y, scoring = 'neg_mean_absolute_error')
#print(scores)
#predicted_price = forest_model.predict(test_X)
predicted_price = data_pipeline.predict(test_X)
print(predicted_price)


submission_file = pd.DataFrame({'Id' : test_data.Id, 'SalePrice' : predicted_price})
submission_file.to_csv('submission.csv', index = False)




#predicted_value = forest_model.predict(test_X)
#print(mean_absolute_error(test_y, predicted_value))

#melbourne_data_adress = melbourne_data.Address
#print(melbourne_data_adress.head())

#melbourne_columns = ['Rooms', 'Price']
#two_columns = melbourne_data[melbourne_columns]
#two_columns.describe()

#main_file_path = '../input/train.csv'
#data = pd.read_csv(main_file_path)


 