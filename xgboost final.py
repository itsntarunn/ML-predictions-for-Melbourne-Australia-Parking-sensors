import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import xgboost as xgb

def train_xgb_model(X_train, y_train, X_val, y_val, X_test, y_test, datetime_test, device_ids_test, duration_minutes_test, scaler=None, param_grid=None):
    """
    Train an XGBoost model on the provided data with hyperparameter tuning using GridSearchCV,
    and evaluate on the test set.
    
    Parameters:
    - X_train, X_val, X_test: Training, validation, and test features (numpy arrays)
    - y_train, y_val, y_test: Training, validation, and test labels (numpy arrays)
    - datetime_test: DataFrame with datetime columns for test data (used for saving results)
    - device_ids_test: Series with device IDs for test data (used for saving results)
    - duration_minutes_test: Series with duration in minutes for test data (used for saving results)
    - scaler: Scaler object to normalize features (default=None, assumes features are already normalized)
    - param_grid: Dictionary of XGBoost parameters for GridSearchCV (default=None, uses a predefined grid if not provided)
    
    Returns:
    - Dictionary with evaluation metrics (MAE, RMSE, Accuracy)
    - DataFrame with predictions and probabilities
    """
    if param_grid is None:
        param_grid = {
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'gamma': [0, 0.1, 0.2],
            'eta': [0.01, 0.1, 0.3]
        }

    if scaler is None:
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

    # Convert data to DMatrix format for XGBoost
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    dval = xgb.DMatrix(X_val_scaled, label=y_val)
    dtest = xgb.DMatrix(X_test_scaled, label=y_test)

    # Initialize XGBoost model
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', seed=42)

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=2)
    grid_search.fit(X_train_scaled, y_train)

    # Get the best parameters and train the final model
    best_params = grid_search.best_params_
    print(f"Best parameters found: {best_params}")

    # Train the XGBoost model with the best parameters
    model = xgb.train(best_params, dtrain, num_boost_round=1000, evals=[(dval, 'validation')], early_stopping_rounds=20, verbose_eval=False)

    # Make predictions on the test set
    y_test_prob = model.predict(dtest)
    y_test_pred = (y_test_prob > 0.5).astype(int)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"XGBoost Test MAE: {mae}")
    print(f"XGBoost Test RMSE: {rmse}")
    print(f"XGBoost Test Accuracy: {accuracy}")

    # Combine predictions and probabilities with the original data
    result_data = pd.DataFrame({
        'DeviceId': device_ids_test.reset_index(drop=True),
        'ArrivalTime': datetime_test['ArrivalTime'].reset_index(drop=True),
        'DepartureTime': datetime_test['DepartureTime'].reset_index(drop=True),
        'DurationMinutes': duration_minutes_test.reset_index(drop=True),
        'VehiclePresent': y_test.reset_index(drop=True),
        'Prediction': y_test_pred,
        'Probability': y_test_prob
    })

    return {'MAE': mae, 'RMSE': rmse, 'Accuracy': accuracy}, result_data

# Example usage:

# Step 1: Load Data
file_path = r'C:\Users\langi longi\Desktop\New folder\combined_sorted_data_13157combined.xlsx'
data = pd.read_excel(file_path)

# Convert 'ArrivalTime' and 'DepartureTime' to datetime
data['ArrivalTime'] = pd.to_datetime(data['ArrivalTime'])
data['DepartureTime'] = pd.to_datetime(data['DepartureTime'])

# Sort data based on the timestamp (assuming 'ArrivalTime' indicates the sequence)
data.sort_values(by='ArrivalTime', inplace=True)

# Convert 'DurationSeconds' column to integer type and create 'DurationMinutes'
data['DurationSeconds'] = data['DurationSeconds'].astype(int)
data['DurationMinutes'] = data['DurationSeconds'] / 60

# Extract additional features from 'ArrivalTime'
data['ArrivalHour'] = data['ArrivalTime'].dt.hour
data['ArrivalDayOfWeek'] = data['ArrivalTime'].dt.dayofweek
data['ArrivalIsWeekend'] = (data['ArrivalTime'].dt.dayofweek >= 5).astype(int)

# Convert 'Vehicle Present' to binary
data['VehiclePresent'] = data['Vehicle Present'].astype(int)

# Define Features and Target
X = data.drop(columns=['VehiclePresent', 'Vehicle Present'])
y = data['VehiclePresent']

# Save the datetime columns and additional info to be used later for saving results
datetime_cols = X[['ArrivalTime', 'DepartureTime']]
device_ids = data['DeviceId']
duration_minutes = data['DurationMinutes']

# Drop the datetime columns from X
X = X.drop(columns=['ArrivalTime', 'DepartureTime'])

# Step 2: Initial Split (90% train/validation, 10% test)
X_train_val, X_test, y_train_val, y_test, datetime_train_val, datetime_test, device_ids_train_val, device_ids_test, duration_minutes_train_val, duration_minutes_test = train_test_split(
    X, y, datetime_cols, device_ids, duration_minutes, test_size=0.1, shuffle=False)

# Step 3: Second Split (70% train, 20% validation of the original dataset)
X_train, X_val, y_train, y_val, datetime_train, datetime_val, device_ids_train, device_ids_val, duration_minutes_train, duration_minutes_val = train_test_split(
    X_train_val, y_train_val, datetime_train_val, device_ids_train_val, duration_minutes_train_val, test_size=0.2222, shuffle=False)  # 0.2222 * 0.9 = 0.2

# Display the number of data points in each set
print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Define the parameter grid for grid search
param_grid = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2],
    'eta': [0.01, 0.1, 0.3]
}

# Train and evaluate XGBoost model with grid search using the function
evaluation_metrics, result_data = train_xgb_model(X_train.values, y_train.values,
                                                  X_val.values, y_val.values,
                                                  X_test.values, y_test,
                                                  datetime_test,
                                                  device_ids_test, duration_minutes_test,
                                                  param_grid=param_grid)

# Save predictions and probabilities to Excel
predictions_file = r'C:\Users\langi longi\Desktop\New folder\xgb_predictions_grid_search.xlsx'
result_data.to_excel(predictions_file, index=False)
print(f"Predictions and probabilities saved to {predictions_file}")

# Print evaluation metrics
print(f"Evaluation Metrics:")
print(f" - MAE: {evaluation_metrics['MAE']}")
print(f" - RMSE: {evaluation_metrics['RMSE']}")
print(f" - Accuracy: {evaluation_metrics['Accuracy']}")
