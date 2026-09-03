with open("labels.txt") as file:
    lines = file.readlines()

for line in lines:
    values = line.strip().split()

    class_id = int(values[0])
    x_center = float(values[1])
    y_center = float(values[2])
    width = float(values[3])
    height = float(values[4])

    print("Class:", class_id)
    print("Center:", x_center, y_center)
    print("Width:", width)
    print("Height:", height)
    print()