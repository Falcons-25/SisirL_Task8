test = [
    "25,12.77,79.2,45,77,23.45,0,0,2,25,5,12,241,25,100,12.78,77.22,0.82,-150,-57,45,12,99,65,12.78,77.21,0.82",
    "24,12.77,79.21,40,78,23.44,0,0,1,35,50,5,230,10,100,12.78,77.22,0.82",
    "25,12.77,79.21,20,77,23.44,0,0,3,0,0,35,104,45,55,12.78,77.22,0.9,-104,56,5,55,55,55,12.78,77.23,0.81,102,-96,36,84,56,24,12.77,77.21,0.8",
]

for i in test:
    parts = i.split(',')
    d = {
    'latitude': float(parts[1]),
    'longitude': float(parts[2]),
    'heading': float(parts[3]),
    'altitude': float(parts[0]),
    'speed': float(parts[4]),
    'battery': float(parts[5]),
    'pitch':float(parts[6]),
    'bank_angle':float(parts[7]),
    'number_of_circles':int(parts[8]),
    'circles':[tuple(parts[9*i:9*(i+1)]) for i in range(1, len(parts)//9)]
    }
    print(d)
    print()