points = [
    [0, 0],
    [100, 0],
    [100, 100],
    [0, 100]
]


H = [
    [1, 0.2, 0],
    [0.1, 1, 0],
    [0.001, 0.002, 1]
]

def to_homogenous(point):
    return [point[0],point[1],1]

def multiply_matrix_point(matrix, point):
    result = [0, 0, 0]
    for i in range(3):
        total = 0
        for j in range(3):
            total += matrix[i][j] * point[j]
        result[i] = total
    return result

def to_cartesian(point):
    X, Y, W = point
    if W == 0:
        return None
    return [X / W, Y / W]

def apply_homography(points,H):
    hp = to_homogenous(points)
    transformed = multiply_matrix_point(H, hp)
    return to_cartesian(transformed)

print("Original Points:")

for point in points:
    print(point)


print("\nTransformed Points:")
for point in points:
    result = apply_homography(point, H)
    print(result)
    