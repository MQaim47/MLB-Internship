import numpy as np

def build_matrix(points1, points2):
    A = []
    for i in range(len(points1)):
        x, y = points1[i]
        u, v = points2[i]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    return np.array(A,dtype=float)

def estimate_homography(points1, points2):
    if len(points1)<4 or len(points2)<4:
        raise ValueError("At least 4 points are required to estimate homography.")

    A = build_matrix(points1, points2)
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H = h.reshape(3, 3)
    return H / H[2, 2]  

def to_homogenous(point):
    return np.array([point[0], point[1], 1])

def apply_homography(points,H):
    p=to_homogenous(points)
    transformed=H@p
    transformed/=transformed[2]
    return transformed[:2]



src_points = np.array([
    [0, 0],
    [100, 0],
    [100, 100],
    [0, 100]
], dtype=float)

dst_points = np.array([
    [10, 20],
    [120, 10],
    [130, 140],
    [20, 130]
], dtype=float)


H = estimate_homography(
    src_points,
    dst_points
)

print("Estimated Homography Matrix:\n")
print(H)

print("\nVerification:\n")


for point in src_points:
    transformed_point = apply_homography(point, H)
    print(f"Original: {point}, Transformed: {transformed_point}")