import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

# Load training data from all stores
train_data = pd.read_csv("data/train.csv", parse_dates=['date'])

# Load holidays data
holidays_data = pd.read_csv("data/holidays_events.csv", parse_dates=['date'])

# Load oil prices data
oil_data = pd.read_csv("data/oil.csv", parse_dates=['date'])

# Print initial oil data to inspect
print("Initial oil data with potential missing values:")
print(oil_data.head(20))

# Assuming the correct column name is 'dcoilwtico'
oil_data.rename(columns={'dcoilwtico': 'oil_price'}, inplace=True)

# Create a complete date range from the minimum to maximum date in train_data
date_range = pd.date_range(start=train_data['date'].min(), end=train_data['date'].max())

# Create a DataFrame with the complete date range
complete_dates = pd.DataFrame(date_range, columns=['date'])

# Merge the complete date range with oil_data
complete_oil_data = pd.merge(complete_dates, oil_data, on='date', how='left')

# Fill missing values in oil prices using a rolling window mean
complete_oil_data['oil_price'] = complete_oil_data['oil_price'].interpolate(method='linear')
complete_oil_data['oil_price'] = complete_oil_data['oil_price'].fillna(complete_oil_data['oil_price'].rolling(window=10, min_periods=1, center=True).mean())

# Verify that the missing values have been handled
print("Complete oil data after filling missing values:")
print(complete_oil_data.isna().sum())

# Filter holidays to only include rows where type is 'Holiday'
holidays_data = holidays_data[holidays_data['type'] == 'Holiday']

# Add holiday feature
holidays_data['is_holiday'] = 1
train_data = pd.merge(train_data, holidays_data[['date', 'is_holiday']], on='date', how='left')
train_data['is_holiday'] = train_data['is_holiday'].fillna(0)

# Merge train_data with the complete oil data
train_data = pd.merge(train_data, complete_oil_data[['date', 'oil_price']], on='date', how='left')

# Fill any remaining missing values in the merged train_data using the mean of the surrounding 10 values
train_data['oil_price'] = train_data['oil_price'].fillna(train_data['oil_price'].rolling(window=10, min_periods=1, center=True).mean())

# Check for missing values after merge
print("train_data after merging oil prices:")
print(train_data.isna().sum())

# Only data from store number 1
s1_df = train_data[train_data['store_nbr'] == 1]

# Group by date to get total sales and total onpromotion for store 1
s1_total_sales_df = s1_df.groupby('date').agg({
    'sales': 'sum',
    'onpromotion': 'sum',
    'is_holiday': 'max',
    'oil_price': 'mean'
}).reset_index()

# Check for nan or infinite values in the dataset
print("s1_total_sales_df missing values and infinities:")
print(s1_total_sales_df.isna().sum())
print(np.isinf(s1_total_sales_df).sum())

plt.figure(figsize=(10, 6))
plt.plot(s1_total_sales_df['date'], s1_total_sales_df['sales'])
plt.title('Sales Data')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(s1_total_sales_df[['sales', 'onpromotion', 'is_holiday', 'oil_price']])

# Verify that the scaled data does not contain nan or infinite values
print("Scaled data missing values and infinities:")
print(np.isnan(scaled_data).sum())
print(np.isinf(scaled_data).sum())

# Split the data into training and test sets
train_size = int(len(scaled_data) * 0.8)
train, test = scaled_data[:train_size], scaled_data[train_size:]

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, 0]) # Target is the 'sales' column
    return np.array(X), np.array(Y)

look_back = 10
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))

# Create and compile the LSTM model
model = Sequential()
model.add(Input(shape=(look_back, trainX.shape[2])))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

# Use Adam optimizer with a lower learning rate
optimizer = Adam(learning_rate=0.001, clipvalue=1.0)
model.compile(loss='mean_squared_error', optimizer=optimizer)

# Train the model with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10)

model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2, validation_data=(testX, testY), callbacks=[early_stop])

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Invert predictions to get actual values
sales_scaler = MinMaxScaler(feature_range=(0, 1))
sales_scaler.min_, sales_scaler.scale_ = scaler.min_[0], scaler.scale_[0]  # Use the scaler parameters for 'sales'
trainPredict = sales_scaler.inverse_transform(trainPredict)
trainY = sales_scaler.inverse_transform([trainY])
testPredict = sales_scaler.inverse_transform(testPredict)
testY = sales_scaler.inverse_transform([testY])

# Calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print(f'Train Score: {trainScore:.2f} RMSE')
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print(f'Test Score: {testScore:.2f} RMSE')

# Initialize arrays for plotting
trainPredictPlot = np.empty((len(scaled_data), 1))
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

testPredictPlot = np.empty((len(scaled_data), 1))
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(scaled_data) - 1, :] = testPredict

plt.figure(figsize=(10, 6))
plt.plot(s1_total_sales_df['date'], scaler.inverse_transform(scaled_data)[:, 0], label='Actual Sales')
plt.plot(s1_total_sales_df['date'], trainPredictPlot, label='Train Predict')
plt.plot(s1_total_sales_df['date'], testPredictPlot, label='Test Predict')
plt.title('Sales Prediction')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

print(train_data.head())
print(s1_df.tail(3))
print(s1_total_sales_df.tail(3))
