import pandas as pd
import numpy as np
from pynput.keyboard import Listener, Key
import time
from tensorflow.keras.models import load_model
import joblib

model = load_model('keystroke_model.h5')

label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

sequence_length = 5 
last_intervals = []
keystroke_data = []

def process_and_predict(last_intervals, model, scaler, label_encoder):
    if len(last_intervals) < sequence_length:
        return None, None

    sequence = np.array(last_intervals[-sequence_length:]).reshape(1, sequence_length, -1)
    sequence_normalized = scaler.transform(sequence.reshape(-1, 1)).reshape(1, sequence_length, -1)

    prediction = model.predict(sequence_normalized)
    predicted_key = label_encoder.inverse_transform([int(round(prediction[0, 0]))])[0]
    predicted_interval = prediction[0, 1]

    return predicted_key, predicted_interval

def collect_keystroke_data():
    last_time = None

    def on_press(key):
        nonlocal last_time
        if key == Key.enter:
            return  

        current_time = time.time()
        if last_time is not None:
            interval = current_time - last_time
            last_intervals.append(interval)

            if len(last_intervals) == sequence_length:
                predicted_key, predicted_interval = process_and_predict(last_intervals, model, scaler, label_encoder)
                keystroke_data.append({'Actual Key': str(key), 'Actual Interval': interval,
                                       'Predicted Key': predicted_key, 'Predicted Interval': predicted_interval})
                
                print(f"Actual: {str(key)}, {interval} | Predicted: {predicted_key}, {predicted_interval}")

        last_time = current_time

    with Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == "__main__":
    collect_keystroke_data()
    df_keystrokes = pd.DataFrame(keystroke_data)
    print(df_keystrokes)  # Print the entire DataFrame
