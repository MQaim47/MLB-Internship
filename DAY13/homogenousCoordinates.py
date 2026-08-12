import numpy as np

def to_homogenous(point):
    x,y = point
    return np.array([x,y,1],dtype=float)

def to_cartesian(point):
    X,Y,W=point

    if W==0:
        return None

    return np.array([X/W,Y/W])

# def transform_point(point,matrix):
#     homogenous_point = to_homogenous(point)

#     transformed = matrix @ homogenous_point
#     return to_cartesian(transformed)

def matrix_multiply(matrix, point):

    result = [0, 0, 0]

    for i in range(3):

        total = 0

        for j in range(3):
            total += matrix[i][j] * point[j]

        result[i] = total

    return result
points = [
    (100, 50),
    (200, 100),
    (300, 150)
]

tx=50
ty=30

translation_matrix = np.array([
    [1, 0, tx],
    [0, 1, ty],
    [0, 0, 1]
])

print("Original Points:")
for point in points:
    print(point)

print("\nTransformed Points:")
for point in points:
    result = matrix_multiply(point, translation_matrix)
    print(result)