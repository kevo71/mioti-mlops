"""This program loads a gradient boosting model to predict the health status
of a user. It launches an API and then you can post requests in the foloowing 
format.

Dictionary required format for data input to model:

{
    'day': str (format='%Y-%m-%d'),
    'sex': int (1 = male, 0 = female),
    'age': int,
    'weight': float,
    'hour': int (0-23),
    'bpml': int,
    'day_of_week': int
}

Model will return a health status: 0 = OK, 1 = Bad.

Sample data inputs:

{
    "day": "2007-12-29",
    "sex": 0,
    "age": 30,
    "weight": 81.7,
    "hour": 13,
    "bpm": 166,
    "day_of_week": "Saturday"
} 

returns "status" = 1, Bad

{
    "day": "2007-12-31",
    "sex": 1,
    "age": 63,
    "weight": 62.9,
    "hour": 1,
    "bpm": 100,
    "day_of_week": "Monday"
} 

returns "status" = 0, OK

"""


from fastapi import FastAPI
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

model_file_name = 'gb_model.sav'

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_file_path = os.path.join(script_dir, model_file_name)
except NameError:
    model_file_path = model_file_name

model = joblib.load(model_file_path)
scaler = joblib.load('scaler.sav') # need fitted scaler as well as model

app = FastAPI()


def transform_input_data(input_data):
    # columns expected by model
    expected_columns = ['sex', 'age', 'weight', 'bpm',
                    'hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 
                    'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23', 
                    'day_of_week_Friday', 'day_of_week_Monday', 'day_of_week_Saturday', 'day_of_week_Sunday', 'day_of_week_Thursday', 'day_of_week_Tuesday', 'day_of_week_Wednesday']
    
    # convert input data to DataFrame
    df = pd.DataFrame([input_data])
    df['day'] = pd.to_datetime(df['day'], format='%Y-%m-%d')
    df["day_of_week"] = df["day"].dt.day_name()
    
    # drop 'status' column if it exists
    df = df.drop(columns=['status'], errors='ignore')
    
    # one-hot encode 'hour' and 'day_of_week' columns
    df = pd.get_dummies(df, columns=['hour', 'day_of_week'])
    
    # ensure all expected columns are present, adding missing ones as zeros
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    # reorder columns to match the expected format
    df = df[expected_columns]
    
    # normalize the required columns
    columns_to_normalize = ['age', 'weight', 'bpm']
    df[columns_to_normalize] = scaler.transform(df[columns_to_normalize])  

    return df


@app.get("/")
def read_root():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: dict):
    transformed_data = transform_input_data(data)
    prediction = model.predict(transformed_data)
    return {"prediction": prediction.tolist()}
