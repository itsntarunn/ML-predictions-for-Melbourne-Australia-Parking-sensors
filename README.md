# Machine Learning Predictions and Sensor Pairing for Melbourne, Australia

This repository contains Python scripts for analyzing sensor data and predicting vehicle presence in Melbourne, Australia using machine learning. The project utilizes XGBoost and GridSearchCV for hyperparameter tuning to optimize the model’s performance. Additionally, sensor pairing is performed to optimize the use of available data for better predictions.

## Files

- **model.py**: Contains the implementation of the XGBoost model with hyperparameter tuning and evaluation.
- **data_preprocessing.py**: Handles data cleaning, feature extraction, and preprocessing.
- **sensor_pairing.py**: Logic for sensor pairing to improve prediction accuracy.
  
## Data

The data used in this project can be accessed from the following sources:
- [Melbourne Open Data Portal](https://data.melbourne.vic.gov.au/)
- [Australian Government Open Data](https://data.gov.au/)

### Example Data Structure:
The data contains columns such as:
- **ArrivalTime**: Timestamp of when the vehicle arrives.
- **DepartureTime**: Timestamp of when the vehicle leaves.
- **DeviceId**: Unique identifier for the sensor devices.
- **Vehicle Present**: Binary column indicating if a vehicle was detected (1 for present, 0 for absent).
- **DurationSeconds**: Duration the vehicle was present, in seconds.

## Installation

### Requirements:
1. Python 3.x
2. The following libraries:
    - pandas
    - numpy
    - scikit-learn
    - xgboost
    - openpyxl

### Install dependencies:
```bash
pip install -r requirements.txt
