import pandas as pd
from pydantic import BaseModel
import joblib
from fastapi import FastAPI, HTTPException
from typing import List
from tsfresh import extract_features


# Load the feature list from CSV file
feature_list = pd.read_csv("../data/model_features.csv")['Feature'].tolist()

# Create a Pydantic model dynamically based on the feature list
class HydraulicSystemMonitoring(BaseModel):
    pass

# Add the features dynamically to the hydraulic system monitoring model
for feature in feature_list:
    setattr(HydraulicSystemMonitoring, feature, float)

# Load the trained model
model = joblib.load("../data/trained_model.joblib")

# Create the FastAPI app
app = FastAPI()


def loading_data():
    # Read each file into a DataFrame
    df_PS2 = pd.read_csv("./data_subset/PS2.txt", sep="\t", header=None)
    df_FS1 = pd.read_csv("./data_subset/FS1.txt", sep="\t", header=None)
    df_profile = pd.read_csv("./data_subset/profile.txt", sep="\t", header=None)
    
    df_profile.columns = ['cooler_condition_%', 'valve_condition_%', 'internal_pump_leakage',
                          'hydraulic_accumulator_bar', 'stable_flag' ]
    return df_PS2, df_FS1, df_profile


def resample_dataframes(df_PS2, df_FS1):
    # Transpose the FS1 DataFrame to have time series as rows
    df_FS1_transposed = df_FS1.T

    # Create a datetime index for the transposed FS1 DataFrame
    fs1_time_index = pd.date_range(start='2024-01-01', periods=len(df_FS1_transposed), freq='10ms')
    df_FS1_transposed.index = fs1_time_index

    # Resample FS1 to match the frequency of PS2 (from 10 Hz to 100 Hz)
    df_FS1_resampled_transposed = df_FS1_transposed.resample('10ms').interpolate()

    # Transpose back to the original format
    df_FS1_resampled = df_FS1_resampled_transposed.T

    # Transpose the PS2 DataFrame to have time series as rows
    df_PS2_transposed = df_PS2.T

    # Create a datetime index for the transposed PS2 DataFrame
    ps2_time_index = pd.date_range(start='2024-01-01', periods=len(df_PS2_transposed), freq='10ms')
    df_PS2_transposed.index = ps2_time_index

    # Resample PS2 to match the frequency of FS1 (from 100 Hz to 10 Hz)
    df_PS2_resampled_transposed = df_PS2_transposed.resample('100ms').interpolate()

    # Transpose back to the original format
    df_PS2_resampled = df_PS2_resampled_transposed.T
    return df_PS2_resampled, df_FS1_resampled

def wrangling_for_tsfresh(df):
    # Reset index to convert the timestamps to a regular column
    df = df.reset_index()
    
    # Melt the DataFrame to create the 'id', 'timestamp', and 'value' columns
    df_melted = pd.melt(df, id_vars=['index'], var_name='timestamp', value_name='value')
    
    # Rename the columns
    df_melted.columns = ['id', 'timestamp', 'value']
    
    # Sort by 'id' for better clarity
    df_melted = df_melted.sort_values(by='id').reset_index(drop=True)
    return df_melted

def preprocessing(df):
    df_extracted_features = extract_features(df, column_id="id", column_sort="timestamp", n_jobs=4)
    return df_extracted_features


# Expose the prediction functionality
@app.post('/predict')
def predict_presence(cycles: List[HydraulicSystemMonitoring]):
    if not cycles:
        raise HTTPException(status_code=400, detail="Input list is empty")
    
    # Perform all preprocessing steps
    df_PS2, df_FS1, df_profile = loading_data()
    df_PS2_resampled, df_FS1_resampled = resample_dataframes(df_PS2, df_FS1)
    df_PS2_melted = wrangling_for_tsfresh(df_PS2_resampled)
    df_FS1_melted = wrangling_for_tsfresh(df_FS1_resampled)
    df_full = pd.concat([df_FS1_melted.rename(columns={'value': 'fs1'}),
                        df_PS2_melted.rename(columns={'value': 'ps2'})['ps2']], axis=1)
    df_input = preprocessing(df_full)
    
    # Convert input data to DataFrame and perform data wrangling
    df_input = pd.DataFrame([[getattr(cycle, feature) for feature in feature_list] for cycle in cycles], columns=feature_list)
    
    # Make predictions using the trained model
    predictions = model.predict(df_input)
    probabilities = model.predict_proba(df_input)[:, 1]
    
    # Format the results
    results = [{'prediction': 'NOT Optimal Valve' if pred == 0 else 'Optimal Valve', 'probability': prob} for pred, prob in zip(predictions, probabilities)]
    
    return results
