# Circle Detection and Calculation of Circle GPS

## Overview
This project provides tools to detect circles using image processing and compute the GPS coordinates of these detected circles based on telemetry data such as altitude, aircraft GPS location, and aircraft pitch and bank angles. The project consists of three main files: `Circle_detection.py`, `Circle_detection_test.py`, and `Telemetry_Standard.py`.

## Files

### 1. Circle_detection.py
This is the primary file responsible for detecting circles and calculating the GPS location of the circle.

- **Inputs**: Altitude, aircraft GPS location, and aircraft pitch/bank angle.
- **Outputs**:
  - GPS location of the detected circle.
  - Circle color.
  - Confidence level that the detected object is a circle.

The algorithm uses computer vision techniques to identify circles and combines telemetry data to compute their precise GPS coordinates.

### 2. Circle_detection_test.py
This file generates random values for the required inputs of `Circle_detection.py`.  
The purpose is to simulate the inputs without connecting to hardware, allowing for easy testing of the circle detection and GPS calculation algorithms.

### 3. Telemetry_Standard.py
This file outlines the telemetry string format that `Circle_detection.py` will generate.  
It describes what each part of the telemetry string conveys, providing a clear standard for the data output.  
The same string processing algorithm can be used to decode and present the telemetry data on a frontend display.

## Dependencies
To run this project, the following libraries are required:

- [OpenCV-Python](https://pypi.org/project/opencv-python/)
- [Arduino](https://www.arduino.cc/en/software)
- [NumPy](https://numpy.org/)

Make sure to install these dependencies using `pip`:

```bash
pip install opencv-python numpy
