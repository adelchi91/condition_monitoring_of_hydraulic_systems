import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from tsfresh import extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import joblib
import logging
from utils.helpers import (loading_data, resample_dataframes,
                            wrangling_for_tsfresh,train_test_split_,
                            preprocessing,
                            validation_score,
                            model_pipeline_trained,
                            cross_validation
)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define file paths
X_train_test_tsf_path = "./data/X_train_test_tsf.csv"
X_val_tsf_path = "./data/X_val_tsf.csv"
model_path = "./data/trained_model.joblib"


def main():
    # loading data
    df_PS2, df_FS1, df_profile = loading_data()
    # resampling data with same frequency 
    df_PS2_resampled, df_FS1_resampled = resample_dataframes(df_PS2, df_FS1)
    # data wrangling - for tsfresh 
    df_PS2_melted = wrangling_for_tsfresh(df_PS2_resampled)
    df_FS1_melted = wrangling_for_tsfresh(df_FS1_resampled) #[lambda x: x.id==0]
    # full dataframe 
    df_full = pd.concat([df_FS1_melted.rename(columns={'value': 'fs1'}),
                        df_PS2_melted.rename(columns={'value': 'ps2'})['ps2']], axis=1)
    # Changing target variable into a binary variable 100=1, the rest=0
    df_profile["valve_condition_%"] = np.where(df_profile["valve_condition_%"] == 100, 1, 0)
    # train-test split
    X_train_test, X_val, y_train_test, y_val = train_test_split_(df_full, df_profile["valve_condition_%"], nb_cycles=2000)
    # preprocessing X_train_test and X_val with tsfresh
    if os.path.exists(X_train_test_tsf_path) and os.path.exists(X_val_tsf_path):
        # Load the data
        X_train_test_tsf = pd.read_csv(X_train_test_tsf_path, index_col=0)
        X_val_tsf = pd.read_csv(X_val_tsf_path, index_col=0)
        logging.info("TSFRESH files loaded successfully.")
        # storing preprocessed data
        X_train_test_tsf.to_csv(X_train_test_tsf_path)
        X_val_tsf.to_csv(X_val_tsf_path)
    else:
        X_train_test_tsf = preprocessing(X_train_test)
        X_val_tsf = preprocessing(X_val)
        logging.info("TSFRESH files run successfully.")
    # training of the model
    pipeline, xgb_feature_names = model_pipeline_trained(X_train_test_tsf, y_train_test)
    # Convert the list to a DataFrame
    df_features = pd.DataFrame(xgb_feature_names, columns=['Feature'])
    # Save the DataFrame to a CSV file
    df_features.to_csv('./data/model_features.csv', index=False)
    # Save the trained model as a joblib file
    joblib.dump(pipeline, model_path)
    # evaluation on test set
    metrics_df = validation_score(pipeline, X_val_tsf[X_train_test_tsf.columns], y_val, dataset="validation")
    # cross-validation
    results_df = cross_validation(pipeline, X_train_test_tsf, y_train_test)
    # saving cv results
    results_df.to_csv("./data/cv_results.csv")
    return

if __name__ == "__main__":
    main()

