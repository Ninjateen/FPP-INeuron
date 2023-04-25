import sys
import os
import pandas as pd
import numpy as np

sys.path.append('src')
import logging

from logger import logging
from sklearn.ensemble import ExtraTreesRegressor

X_train = pd.read_csv("artifacts\\train.csv")
X_test = pd.read_csv("artifacts\\test.csv")
logging.info("Train & test data read")
# Check multicollinearity using VIF
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
pd.options.display.max_rows = None
vif = vif.sort_values(by = 'VIF', ascending = False)
print(vif)

X_train = X_train.drop('Total_Stops', 1,)
X_test = X_test.drop('Total_Stops', 1,)
logging.info('dropping high vif feature')


# Model Creation - Feature Selection 
model = ExtraTreesRegressor(random_state=18)
model.fit(X_train, Y_train)

# Print of Results
print(model.feature_importances_)

# Transformation into pandas series
feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)

# Most important features
feat_importances.sort_values(ascending=False)

X_train = pd.DataFrame(X_train, 
             columns=['Duration_Total_Hour', 'Airline_Jet Airways', 'day_of_journey', 'month_of_journey',
'Airline_Jet Airways Business', 'Airline_Multiple carriers', 'Airline_Air India',
'Destination_Delhi', 'Dep_Min', 'Arrival_Hour', 'Dep_Hour', 'Arrival_Min', 'Destination_New Delhi',
'Destination_Hyderabad', 'Source_Mumbai', 'Airline_IndiGo', 'Source_Banglore', 'Destination_Cochin',
'Source_Delhi'])
X_train = pd.DataFrame(X_train, 
             columns=['Duration_Total_Hour', 'Airline_Jet Airways', 'day_of_journey', 'month_of_journey',
'Airline_Jet Airways Business', 'Airline_Multiple carriers', 'Airline_Air India',
'Destination_Delhi', 'Dep_Min', 'Arrival_Hour', 'Dep_Hour', 'Arrival_Min', 'Destination_New Delhi',
'Destination_Hyderabad', 'Source_Mumbai', 'Airline_IndiGo', 'Source_Banglore', 'Destination_Cochin',
'Source_Delhi'])

