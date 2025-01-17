import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import os

def train_lstm_model(X_train, y_train, X_val, y_val, X_test, y_test, datetime_test, device_ids_test, duration_minutes_test, scaler=None, epochs=100, batch_size=32):
    """
    Train an LSTM model on the provided data and evaluate on the test set.
    
    Parameters:
    - X_train, X_val, X_test: Training, validation, and test features (numpy arrays)
    - y_train, y_val, y_test: Training, validation, and test labels (numpy arrays)
    - datetime_test: DataFrame with datetime columns for test data (used for saving results)
    - device_ids_test: Series with device IDs for test data (used for saving results)
    - duration_minutes_test: Series with duration in minutes for test data (used for saving results)
    - scaler: Scaler object to normalize features (default=None, assumes features are already normalized)
    - epochs: Number of epochs for training (default=100)
    - batch_size: Batch size for training (default=32)
    
    Returns:
    - Dictionary with evaluation metrics (MAE, RMSE, MAPE)
    - DataFrame with predictions and probabilities
    """
    if scaler is None:
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
    
    # Reshape data for LSTM
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_val_lstm = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')

    # Train the model
    model.fit(X_train_lstm, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val_lstm, y_val), verbose=2)

    # Make predictions on the test set
    y_test_prob = model.predict(X_test_lstm).flatten()
    y_test_pred = (y_test_prob > 0.5).astype(int)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mape = np.mean(np.abs((y_test - y_test_pred) / np.where(y_test != 0, y_test, 1))) * 100

    print(f"LSTM Test MAE: {mae}")
    print(f"LSTM Test RMSE: {rmse}")
    print(f"LSTM Test MAPE: {mape}")

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

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}, result_data

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
X = data.drop(columns=['VehiclePresent', 'Vehicle Present', 'DurationSeconds'])
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

# Train and evaluate LSTM model
evaluation_metrics, result_data = train_lstm_model(X_train.values, y_train.values,
                                                  X_val.values, y_val.values,
                                                  X_test.values, y_test,
                                                  datetime_test, device_ids_test, duration_minutes_test)

# Save predictions and probabilities to Excel
predictions_file = r'C:\Users\langi longi\Desktop\New folder\lstmpredictions.xlsx'
result_data.to_excel(predictions_file, index=False)
print(f"Predictions and probabilities saved to {predictions_file}")

# Print evaluation metrics
print(f"Evaluation Metrics:")
print(f" - MAE: {evaluation_metrics['MAE']}")
print(f" - RMSE: {evaluation_metrics['RMSE']}")
print(f" - MAPE: {evaluation_metrics['MAPE']}")
