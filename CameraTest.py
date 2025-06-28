import cv2
import time
import sys

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open the camera.")
    sys.exit(0)

ret, frame = cap.read()
if not ret:
    print("Could not read frame.")
    sys.exit(0)

print(frame.shape)
frame = cv2.resize(frame, (640, 480))
# print(frame)

lowest = 700
highest = 0
counter = 0
while True:
    start_time = time.perf_counter()
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (640, 480))
    fps = 1 / (time.perf_counter() - start_time)
    if counter>100:
        if fps<lowest:
            lowest = fps
        elif highest<fps:
            highest = fps
    else:
        counter+=1
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2, cv2.LINE_AA)
    cv2.imshow("Camera feed", frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        print("Terminated")
        print(f"Lo: {lowest:.2f}, Hi: {highest:.2f}")
        cv2.destroyAllWindows()
        cap.release()
        break
