import sys
import os
import pandas as pd
import numpy as np

sys.path.append('src')
import logging

from logger import logging

# Load the dataset into a Pandas DataFrame...
try:
    df = pd.read_csv("DATASET\\FP DATASET.csv")
except FileNotFoundError:
    logging.error('Dataset file not found!')

# Clean the Date_of_Journey column...
df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'])
df['year_of_journey'] = df['Date_of_Journey'].dt.year
df['month_of_journey'] = df['Date_of_Journey'].dt.month
df['day_of_journey'] = df['Date_of_Journey'].dt.day
df.drop(columns=['Date_of_Journey'], inplace=True)
logging.info('Date_of_Journey column cleaned.')

# Clean the Dep_Time column...
df["Dep_Hour"] = pd.to_datetime(df["Dep_Time"]).dt.hour
df["Dep_Min"] = pd.to_datetime(df["Dep_Time"]).dt.minute
df.drop(["Dep_Time"], axis = 1, inplace = True)
logging.info('Dep_Time column cleaned.')

# Clean the Duration column...
df["Duration_Total_Hour"] = df["Duration"].str.replace("h", '*1').str.replace(' ', '+').str.replace('m', '/60').apply(eval)
df.drop(columns=['Duration'], inplace=True)
logging.info('Duration column cleaned.')

# Clean the Arrival_Time column...
df["Arrival_Hour"] = pd.to_datetime(df["Arrival_Time"]).dt.hour
df["Arrival_Min"] = pd.to_datetime(df["Arrival_Time"]).dt.minute
df.drop(["Arrival_Time"], axis = 1, inplace = True)
logging.info('Arrival_Time column cleaned.')

# Drop unnecessary columns...
df.drop(['year_of_journey',"Additional_Info","Route"], axis = 1, inplace = True)
logging.info('Unnecessary columns dropped.')

# Convert categorical variables into dummy variables...
df = pd.get_dummies(df, prefix=['Airline', 'Source', 'Destination'], columns=['Airline', 'Source','Destination'])
logging.info('Categorical variables converted into dummy variables.')

# Save the cleaned dataset to a directory in the system...
try:
    os.makedirs('DATASET//cleaned_dataset', exist_ok=True)
    df.to_csv('DATASET//cleaned_dataset//cleaned_dataset.csv', index=False)
    logging.info('Cleaned dataset saved to directory.')
except:
    logging.error('Unable to save cleaned dataset to directory!')

# Print the shape of the cleaned dataset
print(df.shape)
logging.info('Data cleaning completed.')
