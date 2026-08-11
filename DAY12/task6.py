import cv2
import numpy as np


def hamming_distance(desc1, desc2):

    distance = 0

    for a, b in zip(desc1, desc2):

        xor = int(a) ^ int(b)

        while xor:
            distance += xor & 1
            xor >>= 1

    return distance


def brute_force_match(descriptors1, descriptors2):

    matches = []

    for i in range(len(descriptors1)):

        best_distance = float("inf")
        best_index = -1

        for j in range(len(descriptors2)):

            distance = hamming_distance(
                descriptors1[i],
                descriptors2[j]
            )

            if distance < best_distance:

                best_distance = distance
                best_index = j

        matches.append(
            (i, best_index, best_distance)
        )

    return matches


def knn_match(descriptors1, descriptors2, k=2):

    all_matches = []

    for i in range(len(descriptors1)):

        distances = []

        for j in range(len(descriptors2)):

            distance = hamming_distance(
                descriptors1[i],
                descriptors2[j]
            )

            distances.append(
                (j, distance)
            )

        distances.sort(
            key=lambda x: x[1]
        )

        all_matches.append(
            distances[:k]
        )

    return all_matches


def ratio_test(knn_matches, ratio=0.75):

    good_matches = []

    for match_pair in knn_matches:

        if len(match_pair) < 2:
            continue

        best = match_pair[0]
        second = match_pair[1]

        if best[1] < ratio * second[1]:

            good_matches.append(
                best
            )

    return good_matches


def draw_matches(
    img1,
    img2,
    kp1,
    kp2,
    matches
):

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    canvas = np.zeros(
        (
            max(h1, h2),
            w1 + w2,
            3
        ),
        dtype=np.uint8
    )

    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2

    for match in matches:

        idx1 = match[0]
        idx2 = match[1]

        x1, y1 = kp1[idx1].pt
        x2, y2 = kp2[idx2].pt

        x1 = int(x1)
        y1 = int(y1)

        x2 = int(x2) + w1
        y2 = int(y2)

        cv2.circle(
            canvas,
            (x1, y1),
            4,
            (0, 255, 0),
            -1
        )

        cv2.circle(
            canvas,
            (x2, y2),
            4,
            (0, 255, 0),
            -1
        )

        cv2.line(
            canvas,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            1
        )

    return canvas


img1 = cv2.imread(
    "object1.jpg"
)

img2 = cv2.imread(
    "object2.jpg"
)

gray1 = cv2.cvtColor(
    img1,
    cv2.COLOR_BGR2GRAY
)

gray2 = cv2.cvtColor(
    img2,
    cv2.COLOR_BGR2GRAY
)

orb = cv2.ORB_create(
    nfeatures=500
)

kp1, des1 = orb.detectAndCompute(
    gray1,
    None
)

kp2, des2 = orb.detectAndCompute(
    gray2,
    None
)

print("Image 1 Keypoints:", len(kp1))
print("Image 2 Keypoints:", len(kp2))

bf_matches = brute_force_match(
    des1,
    des2
)

print(
    "BF Matches:",
    len(bf_matches)
)

bf_matches.sort(
    key=lambda x: x[2]
)

top_matches = bf_matches[:50]

bf_result = draw_matches(
    img1,
    img2,
    kp1,
    kp2,
    top_matches
)

knn_matches = knn_match(
    des1,
    des2,
    k=2
)

good_matches = ratio_test(
    knn_matches,
    ratio=0.75
)

print(
    "Good Matches After Ratio Test:",
    len(good_matches)
)

ratio_result = draw_matches(
    img1,
    img2,
    kp1,
    kp2,
    good_matches[:50]
)

cv2.imshow(
    "BF Matcher",
    bf_result
)

cv2.imshow(
    "KNN + Ratio Test",
    ratio_result
)

cv2.waitKey(0)
cv2.destroyAllWindows()