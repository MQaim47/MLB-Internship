import math

pairs = [
    ((10,20),(40,60)),
    ((100,150),(300,350)),
    ((50,50),(80,120))
]

for p1,p2 in pairs:
    x1,y1 = p1
    x2,y2 = p2
    manualDistance =(((x2 - x1)**2 + (y2 - y1)**2)**0.5)
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    print(f"Manual Distance between {p1} and {p2}: {manualDistance}")
    print(f"Distance between {p1} and {p2}: {distance}")