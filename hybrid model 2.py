import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def train_arima_model(y_train, y_val):
    """
    Train an ARIMA model using the best parameters found with pmdarima.
    """
    # Find the best ARIMA parameters using auto_arima
    autoarima_model = auto_arima(y_train, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
    best_arima_order = autoarima_model.order
    print(f"Best ARIMA parameters: {best_arima_order}")

    # Fit the ARIMA model with the best parameters on the combined training and validation set
    arima_model = ARIMA(np.concatenate([y_train, y_val]), order=best_arima_order).fit()

    return arima_model

def apply_threshold(predictions, threshold=0.2):
    """
    Apply a threshold to predictions to convert them to binary values.
    """
    return (predictions > threshold).astype(int)

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Step 1: Load Data
file_path = r'C:\Users\langi longi\Desktop\New folder\combined_sorted_data_13157combined.xlsx'
data = pd.read_excel(file_path)

# Convert 'ArrivalTime' and 'DepartureTime' to datetime
data['ArrivalTime'] = pd.to_datetime(data['ArrivalTime'])
data['DepartureTime'] = pd.to_datetime(data['DepartureTime'])

# Sort data based on the timestamp (assuming 'ArrivalTime' indicates the sequence)
data.sort_values(by='ArrivalTime', inplace=True)

# Convert 'DurationSeconds' column to integer type
data['DurationSeconds'] = data['DurationSeconds'].astype(int)

# Extract additional features from 'ArrivalTime'
data['ArrivalHour'] = data['ArrivalTime'].dt.hour
data['ArrivalDayOfWeek'] = data['ArrivalTime'].dt.dayofweek
data['ArrivalIsWeekend'] = (data['ArrivalTime'].dt.dayofweek >= 5).astype(int)
data['TripDuration'] = (data['DepartureTime'] - data['ArrivalTime']).dt.total_seconds()

# Convert 'Vehicle Present' to binary
data['VehiclePresent'] = data['Vehicle Present'].astype(int)

# Define Features and Target
X = data.drop(columns=['VehiclePresent', 'Vehicle Present'])
y = data['VehiclePresent']

# Save the datetime, DeviceId, and DurationSeconds columns to be used later for saving results
datetime_deviceid_duration_cols = X[['ArrivalTime', 'DepartureTime', 'DeviceId', 'DurationSeconds']]

# Drop the datetime, DeviceId, and DurationSeconds columns from X
X = X.drop(columns=['ArrivalTime', 'DepartureTime', 'DeviceId', 'DurationSeconds'])

# Initial Split (90% train/validation, 10% test)
X_train_val, X_test, y_train_val, y_test, datetime_deviceid_duration_train_val, datetime_deviceid_duration_test = train_test_split(
    X, y, datetime_deviceid_duration_cols, test_size=0.1, shuffle=False)

# Second Split (70% train, 20% validation of the original dataset)
X_train, X_val, y_train, y_val, datetime_deviceid_duration_train, datetime_deviceid_duration_val = train_test_split(
    X_train_val, y_train_val, datetime_deviceid_duration_train_val, test_size=0.2222, shuffle=False)  # 0.2222 * 0.9 = 0.2

# Display the number of data points in each set
print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train ARIMA model
arima_model = train_arima_model(y_train, y_val)

# Make predictions on the test set using ARIMA
y_test_pred_arima = arima_model.forecast(steps=len(y_test))

# Flatten the predictions
y_test_pred_arima = y_test_pred_arima.flatten()

# Apply threshold to ARIMA predictions
threshold_arima = 0.2  # Adjust the threshold as needed
y_test_pred_arima_thresholded = apply_threshold(y_test_pred_arima, threshold_arima)

# Calculate MAE, RMSE, and MAPE for ARIMA predictions after applying the threshold
mae_arima_thresholded = mean_absolute_error(y_test, y_test_pred_arima_thresholded)
rmse_arima_thresholded = np.sqrt(mean_squared_error(y_test, y_test_pred_arima_thresholded))
mape_arima_thresholded = calculate_mape(y_test, y_test_pred_arima_thresholded)

print(f"ARIMA Test MAE (Thresholded): {mae_arima_thresholded}")
print(f"ARIMA Test RMSE (Thresholded): {rmse_arima_thresholded}")
print(f"ARIMA Test MAPE (Thresholded): {mape_arima_thresholded}")

# Train XGBoost model
param_grid = {
    'max_depth': [3, 5],
    'min_child_weight': [1, 3],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8],
    'eta': [0.1, 0.3]
}

model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', seed=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_test_prob_xgb = best_model.predict_proba(X_test)[:, 1]
y_test_pred_xgb = (y_test_prob_xgb > 0.5).astype(int)

# Calculate MAE for XGBoost predictions
mae_xgb = mean_absolute_error(y_test, y_test_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_test_pred_xgb))

print(f"XGBoost Test MAE: {mae_xgb}")
print(f"XGBoost Test RMSE: {rmse_xgb}")

def train_hybrid_model(y_test_pred_arima_binary, y_test_prob_xgb, y_test):
    X_test_hybrid = np.vstack((y_test_pred_arima_binary, y_test_prob_xgb)).T
    hybrid_model = RandomForestRegressor(n_estimators=100, random_state=42)
    hybrid_model.fit(X_test_hybrid, y_test)
    y_test_hybrid_pred = hybrid_model.predict(X_test_hybrid)
    return y_test_hybrid_pred

# Prepare ARIMA binary predictions for the hybrid model
y_test_pred_arima_binary = y_test_pred_arima_thresholded

# Train hybrid model
y_test_hybrid_pred = train_hybrid_model(y_test_pred_arima_binary, y_test_prob_xgb, y_test)

# Apply threshold to hybrid model predictions
threshold_hybrid = 0.5
y_test_hybrid_pred_thresholded = apply_threshold(y_test_hybrid_pred, threshold_hybrid)

# Calculate MAE, RMSE, and MAPE for hybrid model predictions after applying the threshold
mae_hybrid_thresholded = mean_absolute_error(y_test, y_test_hybrid_pred_thresholded)
rmse_hybrid_thresholded = np.sqrt(mean_squared_error(y_test, y_test_hybrid_pred_thresholded))
mape_hybrid_thresholded = calculate_mape(y_test, y_test_hybrid_pred_thresholded)

print(f"Hybrid Model Test MAE (Thresholded): {mae_hybrid_thresholded}")
print(f"Hybrid Model Test RMSE (Thresholded): {rmse_hybrid_thresholded}")
print(f"Hybrid Model Test MAPE (Thresholded): {mape_hybrid_thresholded}")

# Ensure that the indices align correctly for datetime_deviceid_duration_test
datetime_deviceid_duration_test.reset_index(drop=True, inplace=True)

# Save results to a DataFrame with correct DeviceId and DurationSeconds
result_data = pd.DataFrame({
    'DeviceId': datetime_deviceid_duration_test['DeviceId'],
    'ArrivalTime': datetime_deviceid_duration_test['ArrivalTime'],
    'DepartureTime': datetime_deviceid_duration_test['DepartureTime'],
    'DurationSeconds': datetime_deviceid_duration_test['DurationSeconds'],
    'VehiclePresent': y_test.reset_index(drop=True),
    'ARIMA_Prediction': y_test_pred_arima,
    'ARIMA_Prediction_Thresholded': y_test_pred_arima_thresholded,
    'XGBoost_Prediction': y_test_pred_xgb,
    'XGBoost_Probability': y_test_prob_xgb,
    'Hybrid_Prediction': y_test_hybrid_pred,
    'Hybrid_Prediction_Thresholded': y_test_hybrid_pred_thresholded
})

# Save results to Excel
results_file = r'C:\Users\langi longi\Desktop\New folder\hybrid_model_predictions.xlsx'
result_data.to_excel(results_file, index=False)
print(f"Results saved to {results_file}")

# Save MAE, RMSE, and MAPE values to a DataFrame
metrics_data = {
    'Model': ['ARIMA', 'XGBoost', 'Hybrid Model'],
    'MAE': [mae_arima_thresholded, mae_xgb, mae_hybrid_thresholded],
    'RMSE': [rmse_arima_thresholded, rmse_xgb, rmse_hybrid_thresholded],
    'MAPE': [mape_arima_thresholded, calculate_mape(y_test, y_test_pred_xgb), mape_hybrid_thresholded]
}
metrics_df = pd.DataFrame(metrics_data)
metrics_file = r'C:\Users\langi longi\Desktop\New folder\hybrid_metrics.xlsx'
metrics_df.to_excel(metrics_file, index=False)
print(f"MAE, RMSE, and MAPE values saved to {metrics_file}")

