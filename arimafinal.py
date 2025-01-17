import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

def train_arima_model(y_train, y_val):
    """
    Train an ARIMA model using the best parameters found with pmdarima.

    Parameters:
    - y_train (pd.Series): Training target variable
    - y_val (pd.Series): Validation target variable

    Returns:
    - arima_model (ARIMA): Fitted ARIMA model
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

    Parameters:
    - predictions (np.ndarray): Array of predictions
    - threshold (float): Threshold value for binary classification

    Returns:
    - binary_predictions (np.ndarray): Binary classified predictions
    """
    return (predictions > threshold).astype(int)

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Parameters:
    - y_true (np.ndarray): True values
    - y_pred (np.ndarray): Predicted values

    Returns:
    - mape (float): Mean Absolute Percentage Error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

if __name__ == "__main__":
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

    # Ensure that the indices align correctly for datetime_deviceid_duration_test
    datetime_deviceid_duration_test.reset_index(drop=True, inplace=True)

    # Save thresholded ARIMA predictions to a DataFrame with correct DeviceId and DurationSeconds
    result_data_thresholded = pd.DataFrame({
        'DeviceId': datetime_deviceid_duration_test['DeviceId'],
        'ArrivalTime': datetime_deviceid_duration_test['ArrivalTime'],
        'DepartureTime': datetime_deviceid_duration_test['DepartureTime'],
        'DurationSeconds': datetime_deviceid_duration_test['DurationSeconds'],
        'VehiclePresent': y_test.reset_index(drop=True),
        'ARIMA_Prediction': y_test_pred_arima,
        'ARIMA_Prediction_Thresholded': y_test_pred_arima_thresholded
    })

    # Save thresholded results to Excel
    results_file_thresholded = r'C:\Users\langi longi\Desktop\New folder\arima_results_thresholded.xlsx'
    result_data_thresholded.to_excel(results_file_thresholded, index=False)
    print(f"Thresholded results saved to {results_file_thresholded}")
