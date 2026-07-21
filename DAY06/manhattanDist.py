import math

pairs = [
    ((10,20),(40,60)),
    ((100,150),(300,350)),
    ((50,50),(80,120))
]

for p1,p2 in pairs:
    x1,y1 = p1
    x2,y2 = p2
    
    euclideanDistance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    manhattanDistance = abs(x2 - x1) + abs(y2 - y1)
    
    print(f"Euclidean Distance between {p1} and {p2}: {euclideanDistance}")
    print(f"Manhattan Distance between {p1} and {p2}: {manhattanDistance}")
    
    