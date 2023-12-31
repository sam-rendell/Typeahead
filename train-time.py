import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

df = pd.read_csv('character_timing_data.csv')

label_encoder = LabelEncoder()
df['Key_Encoded'] = label_encoder.fit_transform(df['Key'])

scaler = StandardScaler()
df['Interval_Normalized'] = scaler.fit_transform(df[['Interval']])

def create_sequences(df, seq_length):
    xs, ys = [], []
    for i in range(len(df) - seq_length):
        x = df.iloc[i:(i + seq_length)].Interval_Normalized.values
        y = df.iloc[i + seq_length].Key_Encoded
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

sequence_length = 5

X, y = create_sequences(df, sequence_length)

X = X.reshape((X.shape[0], X.shape[1], 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
    Dropout(0.2),
    LSTM(50),
    Dense(len(label_encoder.classes_), activation='softmax')  # Classification layer
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

y_train = to_categorical(y_train, num_classes=len(label_encoder.classes_))
y_test = to_categorical(y_test, num_classes=len(label_encoder.classes_))

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

model.summary()
model.save('keystroke_prediction_model.keras')
