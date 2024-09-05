import time, random
import Circle_detection
import threading

current_lat = 13.199875
current_long = 77.702672
current_altitude = 30
current_camera_angle = 0
Circle_detection.setup(90)
thread_detect = threading.Thread(target=Circle_detection.detect_circles)
try:
    thread_detect.start()
except KeyboardInterrupt:
    print("User terminated operation.")
try:
    while True:
        Circle_detection.import_values(current_lat, current_long, current_altitude, current_camera_angle)
        current_lat += random.randint(-5, 5)/10000
        current_long += random.randint(-5, 5)/10000
        alt_increment = random.randint(-5, 5)
        current_altitude = current_altitude + (alt_increment if current_altitude+alt_increment>0 else -alt_increment)
        current_camera_angle = random.randint(-5, 5)
        time.sleep(0.015)
except KeyboardInterrupt:
    print("User terminated operation.")
