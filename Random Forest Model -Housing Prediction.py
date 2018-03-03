DIR = r"C:\Users\Dmob\Desktop\ANDROID DEVELOPMENT\KAGGLE"

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

print(1+91)
# save filepath to variable for easier access
melbourne_file_path = '../input/house-prices-advanced-regression-techniques/train.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path)

#my_imputer = Imputer()
#melbourne_data2 = my_imputer.fit_transform(melbourne_data)

#melbourne_data2 = pd.DataFrame.fillna(melbourne_file_path)
# print a summary of the data in Melbourne data
#print(melbourne_data.describe())
#print(melbourne_data.columns)
y = melbourne_data.SalePrice 
#melbourne_predictors = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
  #                      'YearBuilt', 'Lattitude', 'Longtitude']
melbourne_predictors = ['LotArea','YearBuilt', '1stFlrSF', '2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X = melbourne_data[melbourne_predictors]

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

#print("Making predictions for the following 5 houses:")
#print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))


predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

for max_leaf_nodes in [5,30,40, 50,100,200,500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# Read the data
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

# pull data into target (y) and predictors (X)
train_y = train.SalePrice
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

# Create training predictors data
train_X = train[predictor_cols]

my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)

# Read the test data
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predictor_cols]
# Use the model to make predictions
predicted_prices = my_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)