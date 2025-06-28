import math
import time
import threading
import copy
import serial, serial.serialutil, serial.tools.list_ports
from pymavlink import mavutil
import cv2
import numpy as np
import os, signal

class ParsedData:
    latitude = 0                # +ve N
    longitude = 0               # +ve E
    altitude = 5                # m
    pitch = 0                   # +ve up rad
    bank_angle = 0              # +ve R rad
    heading = 0                 # 0N 90E
    number_of_circles = 0
    circles = []
    speed = 0                   # m/s
    battery = 0                 # V

    def __init__(self, message):
        # print(datastr)
        if message=="":
            self.latitude = 0
            self.longitude = 0
            self.altitude = 5
            self.heading = 0
            self.bank_angle = 0
            self.pitch = 0
            self.number_of_circles = 0
            self.circles = []
            self.battery = 0
            self.speed = 0
        else:
            if message.get_type() == 'GLOBAL_POSITION_INT':
                self.altitude = message.alt / 1000.0
            elif message.get_type() == 'ATTITUDE':
                self.pitch = message.pitch
                self.bank_angle = message.roll
            elif message.get_type() == 'GPS_RAW_INT':
                self.latitude = message.lat / 1e7
                self.longitude = message.lon / 1e7
            elif message.get_type() == "VFR_HUD":
                self.heading = message.heading
            elif message.get_type() in ["BATTERY_STATUS", "SYS_STATUS"]:
                try: self.battery = message.voltages[0]/1000
                except Exception:
                    self.battery = message.voltage_battery/1000
        
    def copy(self):
        return copy.deepcopy(self)
    
    def toString(self):
        telemetry_data = f"{self.altitude},{self.latitude},{self.longitude},{self.heading},{self.speed},{self.battery},{self.pitch},{self.bank_angle},{self.number_of_circles},{','.join(self.circles)}"
        # print(telemetry_data)
        return telemetry_data

def initialiseInputPort(port_type: str, port: str, baudrate: int):
    global curr_data, prev_data, error_code, input_serial
    error_code = 0
    if port_type=="ser":
        input_serial = serial.Serial(port=port, baudrate=baudrate)
        print("Serial communication initialised.")
        prev_data = ParsedData("")
        try:
            while True:
                try:
                    data = input_serial.read_all()
                    prev_data = curr_data.copy()
                except NameError:
                    prev_data = ParsedData("")
                curr_data = ParsedData(data)
        except serial.serialutil.SerialException:
            error_code = 1
            end_execution(error_code)
        except KeyboardInterrupt:
            error_code = 2
            end_execution(error_code)
        except UnicodeDecodeError:
            curr_data = ParsedData("")
    elif port_type=="mav":
        input_serial = mavutil.mavlink_connection(device=port, baud=baudrate)
        input_serial.wait_heartbeat()
        print("MAVLink Connected.")
        prev_data = ParsedData("")
        try:
            while True:
                try:
                    msg = input_serial.recv_match(blocking=True)
                    prev_data = curr_data.copy()
                    curr_data = ParsedData(message=msg)
                except NameError:
                    prev_data = ParsedData("")
                    curr_data = ParsedData(msg)
                except serial.serialutil.SerialException:
                    error_code = 1
                    end_execution(error_code)
                except KeyboardInterrupt:
                    error_code = 2
                    end_execution(error_code)
        except KeyboardInterrupt:
            error_code = 2
            end_execution(error_code)

def initialiseOutputPort(port_type: str, port: str, baudrate: int):
    global curr_data, prev_data, error_code, output_serial
    error_code = 0
    if port_type=="ser":
        output_serial = serial.Serial(port=port, baudrate=baudrate)
        try:
            while True:
                try:
                    output_serial.write(f"{prev_data.toString()}\n")
                    prev_data = curr_data.copy()
                    curr_data.circles[:] = []
                except NameError:
                    prev_data = ParsedData("")
        except serial.serialutil.SerialException:
            error_code = 1
            end_execution(error_code)
        except KeyboardInterrupt:
            error_code = 2
            end_execution(error_code)
        except UnicodeDecodeError:
            curr_data = ParsedData("")
    elif port_type=="mav":
        output_serial = mavutil.mavlink_connection(device=port, baud=baudrate)
        output_serial.wait_heartbeat()
        print("MAVLink Connected.")
        try:
            while True:
                try:
                    # send specific
                    print("Cannot use this port as output.")
                    error_code = 3
                    end_execution(error_code)
                except serial.serialutil.SerialException:
                    error_code = 1
                    end_execution(error_code)
                except KeyboardInterrupt:
                    error_code = 2
                    end_execution(error_code)
        except KeyboardInterrupt:
            error_code = 2
            end_execution(error_code)

def end_execution(code: int) -> None:
    """
    1 - Arduino/Serial port disconnected\n
    2 - User termination\n
    3 - No COM Ports\n
    """
    if code==3:
        # logging.error("No COM Port available.")
        print("No COM Port available.")
    elif code==1:
        # logging.error("Arduino Disconnected.")
        print("Arduino disconnected.")
    elif code==2:
        # logging.error("User terminated operation.")
        print("User terminated operation.")
    elif not error_code:
        print("Unknown exc :)")
    os.kill(os.getpid(), signal.SIGINT)

def calculate_gps(pixel_x: int, pixel_y: int, pitch: float, bank_angle: float):
    # pitch = math, bank_angle: float.radians(pitch)
    global mid_x, mid_y, fov, width, height
    new_data = curr_data.copy()
    old_data = prev_data.copy()
    x_component = new_data.latitude - old_data.latitude
    y_component = new_data.longitude - old_data.longitude
    if x_component==0 and y_component==0:
        heading = old_data.heading
    else:
        try:
            if not y_component:
                heading = math.pi if x_component<0 else 0
            elif x_component>0:
                heading = math.atan(y_component/x_component)
            elif x_component<0:
                heading = -math.atan(y_component/x_component)
        except ZeroDivisionError:
            heading = math.pi/2 * (-1 if x_component<0 else 1)
        except Exception:
            print("Error calculating heading")
            return (None, None)
    theta_x = (pixel_x - mid_x)*(fov)/(width) - bank_angle
    theta_y = (pixel_y - mid_y)*(fov)/(height) + pitch
    distance_earthx = math.tan(theta_x+heading)*new_data.altitude
    distance_earthy = math.tan(theta_y+heading)*new_data.altitude
    # theta_z = math.acos(math.sqrt(1 - math.cos(theta_x)**2 - math.cos(theta_y)**2))
    landz_lat = curr_data.latitude + (distance_earthx/111319.4908)
    landz_long = curr_data.longitude + (distance_earthy/111319.4908/math.cos(math.fabs(new_data.latitude)))
    # print(f"Distance: {math.sqrt(distance_earthx**2 + distance_earthy**2)}m\nD_x: {distance_earthx}\nD_y: {distance_earthy}")
    curr_data.heading = heading
    # print(f"{time.strftime("%H:%M:%S")}  {x_component:.4f} {y_component:.4f} {math.degrees(heading):.2f} {math.degrees(theta_x):.3f} {math.degrees(theta_y):.3f} {new_data.altitude} {distance_earthx:.6f} {distance_earthy:.6f} {new_data.latitude} {landz_lat} {new_data.longitude} {landz_long}")
    return landz_lat, landz_long

def setup(field_of_view: int):
    global cap, frame_shape, detector, curr_data, prev_data, mid_x, mid_y, fov, width, height, confidence_threshold
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # logging.error("Could not open camera.")
        print("Could not open video stream.")
    print("Video Streaming.")
    ret, frame = cap.read()
    if ret:
        print("Test framed.")
        frame_shape = frame.shape
    else:
        # logging.error("Could not read frame.")
        print("Could not read frame.")
        return
    curr_data = ParsedData("12.969357,79.155121,405,662,635,654,190.65,663,791,438,964,433")
    prev_data = ParsedData("")
    width = frame_shape[1]
    height = frame_shape[0]
    mid_x = width//2
    mid_y = height//2
    fov = math.radians(field_of_view)
    # params = cv2.SimpleBlobDetector_Params()
    # params.filterByArea = True
    # params.minArea = 100
    # params.filterByCircularity = True
    # params.minCircularity = 0.8
    # params.filterByConvexity = True
    # params.minConvexity = 0.2
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.1
    # detector = cv2.SimpleBlobDetector_create(params)
    on_confidence_threshold_change(80)

def calculate_avg_hsv(frame):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
    avg_hsv = cv2.mean(masked_hsv, mask=mask)[:3]
    return tuple(map(lambda x: round(x), avg_hsv))

def detect_circles():
    global cap, frame_shape, detector
    prev_time = time.perf_counter()
    with open("Circle_log.csv", "a") as file:
        while True:
            ret, frame = cap.read()
            if not ret:
                # logging.error("Could not read frame.")
                print("Could not read frame.")
                print(time.strftime("%Y%m%d %H:%M:%S"), "Could not read frame.", file=file)
                break

            frame_shape = frame.shape
            keypoints = detector.detect(frame)
            blank = np.zeros((1, 1))
            blobs = cv2.drawKeypoints(frame, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            number_of_blobs = len(keypoints)
            if number_of_blobs:
                circles = np.uint16(np.around(blobs))
                curr_data.circles[:] = []
                for i in circles[0, :]:
                    center, radius = (i[0], i[1]), i[2]
                    if i[0]>width or i[1]>height: continue
                    # identify gps
                    landz_latitude, landz_longitude = calculate_gps(center[0], center[1], 0, 0)
                    colour = cv2.cvtColor(np.uint8([[calculate_avg_hsv(frame)]]), cv2.COLOR_HSV2BGR)[0][0].tolist()
                    # display detected circles
                    cv2.circle(frame, center, radius, colour, 2)
                    cv2.circle(frame, center, 2, colour, 3)
                    colour_text = f"HSV: {colour}"
                    cv2.putText(frame, colour_text, (center[0]-radius, center[1]+radius+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
                    if landz_latitude is None or landz_longitude is None:
                        print("Error calculating GPS")
                        continue
                    gps_text = f"{landz_latitude}, {landz_longitude}"
                    cv2.putText(frame, gps_text, (center[0]-radius, center[1]+radius+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
                    # print(f"{curr_data.latitude} -> {curr_data.landz_latitude}\n{curr_data.longitude} -> {curr_data.landz_longitude}\n")
                    curr_data.circles.extend([i[0], i[1], i[2] + list(colour) + [landz_latitude, landz_longitude, 0]])
                    curr_data.number_of_circles += 1
                    # print(curr_data.circles)
            current_time = time.perf_counter()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            # show images https://www.haiku-os.org/legacy-docs/bebook/TheKernelKit_SystemInfo.html
            fps_text = f"FPS: {fps}"
            cv2.putText(frame, fps_text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Number of blobs: {number_of_blobs}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2, cv2.LINE_AA)
            cv2.imshow("Circles Detected", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # logging.error("User terminated operation.")
                print("User terminated operation.")
                break

def on_confidence_threshold_change(val):
    global confidence_threshold
    confidence_threshold = val / 100.0

def calculate_circle_confidence(circle, frame):
    (x, y, r) = circle
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    mean, std_dev = cv2.meanStdDev(frame, mask=mask)
    std_dev = np.mean(std_dev)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    contour_area = cv2.contourArea(contours[0])
    contour_perimeter = cv2.arcLength(contours[0], True)
    circularity = 4*np.pi*contour_area / (contour_perimeter**2)
    confidence = (1 - std_dev / 255) * circularity
    return confidence

def track_coloured_circle():
    cv2.namedWindow("Landing Zone Identification")
    cv2.createTrackbar("Confidence %", "Landing Zone Identification", int(confidence_threshold*100), 100, on_confidence_threshold_change)
    fps = 0
    start_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: could not read frame.")
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=10, maxRadius=100)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # number_of_circles = len(circles)
            curr_data.number_of_circles = 0
            curr_data.circles[:] = []
            for (x, y, r) in circles:
                if (conf:=calculate_circle_confidence((x, y, r), frame)) >= confidence_threshold:
                    avg_hsv = calculate_avg_hsv(frame)
                    avg_hsv_colour = tuple(avg_hsv)
                    colour_bgr = cv2.cvtColor(np.uint8([[avg_hsv_colour]]), cv2.COLOR_HSV2BGR)[0][0]
                    # print(colour_bgr)
                    cv2.circle(frame, (x, y), r, colour_bgr.tolist(), 2)
                    cv2.circle(frame, (x, y), 2, colour_bgr.tolist(), 3)
                    hsv_text = f"HSV: {avg_hsv}"
                    cv2.putText(frame, hsv_text, (x-r, y+r+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_bgr.tolist(), 2)
                    cv2.putText(frame, f"Confidence: {conf:.2f}", (x-r, y-r-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_bgr.tolist(), 2)
                    landz_latitude, landz_longitude = calculate_gps(x, y, 0, 0)
                    if (landz_latitude is None) or (landz_longitude is None):
                        print("Error calculating GPS.")
                        continue
                    gps_text = f"{landz_latitude:.6f}, {landz_longitude:.6f}"
                    cv2.putText(frame, gps_text, (x-r, y+r+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_bgr.tolist(), 2)
                    curr_data.circles.extend([[x, y, r] + list(colour_bgr.tolist()) + [landz_latitude, landz_longitude, conf]])
                    curr_data.number_of_circles += 1
                    # print(curr_data.circles)
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            start_time = time.time()
            frame_count = 0
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Landing Zone Identification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
    end_execution(2)

if __name__ == "__main__":
    global input_serial, output_serial
    comports = sorted([str(x) for x in serial.tools.list_ports.comports()])
    input_port = ""
    output_port = ""
    baudrate = 57600
    if comports:
        if len(comports)==1:
            print("Only 1 COM Port detected:", comports[0])
            error_code = 3
            end_execution(3)
        print("Select COM Port for input:")
        for (i, port) in enumerate(comports):
            print(f"{i+1}. {port}")
        number = int(input("Enter option: "))
        if number!=-1:
            input_port = comports[number-1].split()[0]
        else: input_port = input("Enter port name: ")
        print("Select COM Port for output:")
        for (i, port) in enumerate(comports):
            if port.split()[0] == input_port: continue
            print(f"{i+1}. {port}")
        number = int(input("Enter option: "))
        if number!=-1:
            output_port = comports[number-1].split()[0]
        else: output_port = input("Enter port name: ")
        s = input(f"Use baudrate {baudrate}?")
        if s not in 'yY':
            baudrate = int(input("Enter new baudrate: "))
            if not baudrate: error_code = 2
        if input_port and output_port and baudrate:
            try:
                input_thread = threading.Thread(target=initialiseInputPort, args=(input_port, baudrate))
                input_thread.start()
                time.sleep(1)
                if input_serial:
                    print("Input port initialised.")
                else:
                    print("Input port could not be initialised.")
                    error_code = 1
                    end_execution(error_code)
            except serial.serialutil.SerialException:
                print("Input Port Issue:")
                error_code = 1
                end_execution(error_code)
            except KeyboardInterrupt:
                print("Input Port Issue:")
                error_code = 2
                end_execution(error_code)
            try:
                output_thread = threading.Thread(target=input_port, args=(output_port, baudrate))
                output_thread.start()
                time.sleep(1)
                if output_serial:
                    print("Output Port initialised.")
                else:
                    print("Output port could not be initialised.")
                    error_code = 1
                    end_execution(error_code)
            except serial.serialutil.SerialException:
                print("Output Port Issue:")
                error_code = 1
                end_execution(error_code)
            except KeyboardInterrupt:
                print("Output Port Issue:")
                error_code = 2
                end_execution(error_code)
        else:
            end_execution(error_code)
    else:
        error_code = 1
        end_execution(error_code)
    try:
        setup(90)
        # detect_circles()
        track_coloured_circle()
    except KeyboardInterrupt:
        error_code = 2
        end_execution(error_code)