import pandas as pd
from sklearn.pipeline import Pipeline
from tsfresh.transformers import RelevantFeatureAugmenter
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tsfresh import extract_features
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold


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


def train_test_split_(X, y, nb_cycles=2000):
    X_train_test, X_val = X[lambda x: x["id"]<=(nb_cycles-1)], X[lambda x: x["id"]>(nb_cycles-1)]
    y_train_test, y_val = y.iloc[:nb_cycles], y.iloc[nb_cycles:]
    return X_train_test, X_val, y_train_test, y_val
    


def pipeline_creation(X_train, y_train):
    # Define the pipeline
    pipeline = Pipeline([
        ('augmenter', RelevantFeatureAugmenter(column_id='id', column_sort='timestamp', n_jobs=3)),
        ('classifier', XGBClassifier(
            # random_state=42,
            # n_estimators=100,  # Number of trees in the ensemble
            # max_depth=2,  # Maximum depth of each tree
            # learning_rate=0.1,  # Step size shrinkage used in update to prevent overfitting
            # min_child_weight=1,  # Minimum sum of instance weight (hessian) needed in a child
            # subsample=0.8,  # Subsample ratio of the training instance
            # colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree
            # reg_alpha=0,  # L1 regularization term on weights
            # reg_lambda=1,  # L2 regularization term on weights
            # gamma=0,  # Minimum loss reduction required to make a further partition on a leaf node of the tree
        ))  # XGBoost classifier
    ])

    # Define the evaluation metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    pipeline.set_params(augmenter__timeseries_container=X_train)
    # Perform time series cross-validation
    cv_results = cross_validate(pipeline, X_train, y_train, cv=TimeSeriesSplit(n_splits=5), scoring=scoring)

    # Compute relevant metrics
    metrics_df = pd.DataFrame({
        'Accuracy': cv_results['test_accuracy'],
        'Precision': cv_results['test_precision'],
        'Recall': cv_results['test_recall'],
        'F1': cv_results['test_f1'],
        'AUC': cv_results['test_roc_auc']
    })
    return pipeline, cv_results, metrics_df



def validation_score(pipeline, X_test, y_test, dataset='validation'):
    # Predict the target variable on the test data
    y_pred = pipeline.predict(X_test)

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Create a DataFrame to store the metrics
    metrics_df = pd.DataFrame({
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1': [f1],
        'AUC': [roc_auc]
    })
    metrics_df.to_csv(f"./data/metrics_df_{dataset}.csv")    
    return metrics_df


def preprocessing(df):
    df_extracted_features = extract_features(df, column_id="id", column_sort="timestamp", n_jobs=4)
    return df_extracted_features


def cross_validation(pipeline, X_train_test_tsf, y_train_test):
    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    # Define scoring metrics
    scoring = {'Train AUC': 'roc_auc', 'Test AUC': 'roc_auc'}

    # Perform time series cross-validation
    cv_results = cross_validate(pipeline, X_train_test_tsf, y_train_test, cv=tscv, scoring=scoring, return_train_score=True)

    # Create a DataFrame to store train and test performances
    results_df = pd.DataFrame({
        'Train AUC': cv_results['train_Train AUC'],
        'Test AUC': cv_results['test_Test AUC']
    })

    # Calculate mean and standard deviation
    results_df.loc['Mean'] = results_df.mean()
    results_df.loc['Std'] = results_df.std()
    return results_df


def model_pipeline_trained(X_train_test_tsf, y_train_test):
    # Define the pipeline with only the first step
    pipeline = Pipeline([
        # ('feature_extractor', TSFreshFeatureExtractor(column_id='id', column_sort='timestamp')),
        # ('feature_selection', SelectFromModel(XGBClassifier(random_state=42), max_features=30)),  # XGBoost model for feature selection
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
        ('variance_threshold', VarianceThreshold()),  # Remove constant features
        ('feature_selection', SelectKBest(score_func=f_classif, k=2)),  # Select top 10 features
        ('classifier', XGBClassifier(random_state=42))
    ])
    pipeline.set_output(transform="pandas")
    # Fit the pipeline to your data and train the classifier
    pipeline.fit(X_train_test_tsf, y_train_test)
    # Get feature names used by the XGBoost model
    xgb_feature_names = pipeline.named_steps['classifier'].get_booster().feature_names
    return pipeline, xgb_feature_names
