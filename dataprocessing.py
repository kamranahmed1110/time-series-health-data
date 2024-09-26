import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load the dataset
data = pd.read_csv('health_time_series_indexed.csv')

# Scaling the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences and labels for LSTM
def create_sequences(data, seq_length=3):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Creating sequences
X, y = create_sequences(scaled_data)

# Splitting into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshaping the input to be [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1]))  # Output layer should match the label dimensions

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
epochs = 100
model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2)

# Save the model
model.save('my_models.h5')
