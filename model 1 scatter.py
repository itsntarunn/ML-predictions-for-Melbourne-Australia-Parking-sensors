import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the Excel file
file_path = r'C:\Users\langi longi\Desktop\New folder\hybrid_model2_predictions.xlsx'
data = pd.read_excel(file_path)

# Convert 'ArrivalTime' to datetime and sort by 'ArrivalTime'
data['ArrivalTime'] = pd.to_datetime(data['ArrivalTime'])
data.sort_values(by='ArrivalTime', inplace=True)

# Initialize lists to store extended data for plotting
timestamps = []
actual_status = []
arima_status = []
xgboost_status = []
hybrid_status = []

# Loop over the rows in the dataframe to create the extended data for plotting
for i, row in data.iterrows():
    start_time = row['ArrivalTime']
    duration = pd.Timedelta(seconds=row['DurationSeconds'])
    end_time = start_time + duration

    # Add the start time and end time for each status change
    timestamps.extend([start_time, end_time])
    
    # For each status, add its value at the start time and 0 at the end time
    actual_status.extend([row['VehiclePresent'], 0])
    arima_status.extend([row['ARIMA_Prediction'], 0])
    xgboost_status.extend([row['XGBoost_Prediction'], 0])
    hybrid_status.extend([row['Hybrid_Prediction_Thresholded'], 0])

# Create a DataFrame for plotting
plot_data = pd.DataFrame({
    'Timestamp': timestamps,
    'Actual': actual_status,
    'ARIMA': arima_status,
    'XGBoost': xgboost_status,
    'Hybrid': hybrid_status
})

# Sort the DataFrame by Timestamp
plot_data.sort_values('Timestamp', inplace=True)

# Create the plot
plt.figure(figsize=(12, 6))

# Plot each status with step lines
plt.step(plot_data['Timestamp'], plot_data['Actual'], where='post', label='Actual Status', color='blue')
plt.step(plot_data['Timestamp'], plot_data['ARIMA'], where='post', label='ARIMA Prediction', color='orange')
plt.step(plot_data['Timestamp'], plot_data['XGBoost'], where='post', label='XGBoost Prediction', color='green')
plt.step(plot_data['Timestamp'], plot_data['Hybrid'], where='post', label='Hybrid Prediction', color='purple')

# Set axis labels and title
plt.xlabel('Arrival Time')
plt.ylabel('Status')
plt.title('Vehicle Status Over Time')

# Rotate x-axis labels for better visibility
plt.xticks(rotation=45)

# Add legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
