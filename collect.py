from pynput.keyboard import Listener, Key
import time
import csv

def collect_keystroke_data():
    keystroke_data = []
    last_time = None
    keystroke_count = 0

    def on_press(key):
        nonlocal last_time, keystroke_count
        if key == Key.enter:
            return  
        if last_time is not None:
            interval = time.time() - last_time
            keystroke_data.append((str(key), interval))
            keystroke_count += 1
            if keystroke_count >= 10000:
                return False 
        last_time = time.time()

    with Listener(on_press=on_press) as listener:
        listener.join()

    with open('character_timing_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Key', 'Interval'])
        writer.writerows(keystroke_data)

if __name__ == "__main__":
    collect_keystroke_data()
