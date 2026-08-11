# import cv2
# import numpy as np


# def hamming_distance(desc1, desc2):

#     distance = 0

#     for a, b in zip(desc1, desc2):

#         xor = int(a) ^ int(b)

#         while xor:
#             distance += xor & 1
#             xor >>= 1

#     return distance


# def knn_match(des1, des2):

#     matches = []

#     for i in range(len(des1)):

#         distances = []

#         for j in range(len(des2)):

#             d = hamming_distance(
#                 des1[i],
#                 des2[j]
#             )

#             distances.append(
#                 (j, d)
#             )

#         distances.sort(
#             key=lambda x: x[1]
#         )

#         matches.append(
#             (i, distances[:2])
#         )

#     return matches


# def ratio_test(matches, ratio=0.75):

#     good = []

#     for query_idx, pair in matches:

#         if len(pair) < 2:
#             continue

#         best = pair[0]
#         second = pair[1]

#         if best[1] < ratio * second[1]:

#             good.append(
#                 (query_idx, best[0])
#             )

#     return good


# def draw_matches(
#         img1,
#         img2,
#         kp1,
#         kp2,
#         matches):

#     h1, w1 = img1.shape[:2]
#     h2, w2 = img2.shape[:2]

#     canvas = np.zeros(
#         (
#             max(h1, h2),
#             w1 + w2,
#             3
#         ),
#         dtype=np.uint8
#     )

#     canvas[:h1, :w1] = img1
#     canvas[:h2, w1:] = img2

#     for idx1, idx2 in matches:

#         x1, y1 = kp1[idx1].pt
#         x2, y2 = kp2[idx2].pt

#         x1 = int(x1)
#         y1 = int(y1)

#         x2 = int(x2) + w1
#         y2 = int(y2)

#         cv2.circle(
#             canvas,
#             (x1, y1),
#             4,
#             (0, 255, 0),
#             -1
#         )

#         cv2.circle(
#             canvas,
#             (x2, y2),
#             4,
#             (0, 255, 0),
#             -1
#         )

#         cv2.line(
#             canvas,
#             (x1, y1),
#             (x2, y2),
#             (255, 0, 0),
#             1
#         )

#     return canvas


# img1 = cv2.imread(
#     "object1.jpg"
# )

# img2 = cv2.imread(
#     "object2.jpg"
# )

# gray1 = cv2.cvtColor(
#     img1,
#     cv2.COLOR_BGR2GRAY
# )

# gray2 = cv2.cvtColor(
#     img2,
#     cv2.COLOR_BGR2GRAY
# )

# orb = cv2.ORB_create(
#     nfeatures=1000
# )

# kp1, des1 = orb.detectAndCompute(
#     gray1,
#     None
# )

# kp2, des2 = orb.detectAndCompute(
#     gray2,
#     None
# )

# print("FEATURE DETECTION ")

# print(
#     "Image 1 Keypoints:",
#     len(kp1)
# )

# print(
#     "Image 2 Keypoints:",
#     len(kp2)
# )

# knn_matches = knn_match(
#     des1,
#     des2
# )

# total_matches = len(knn_matches)

# print("KNN MATCHING")

# print(
#     "Total Matches:",
#     total_matches
# )

# good_matches = ratio_test(
#     knn_matches,
#     ratio=0.75
# )

# print(
#     "Good Matches:",
#     len(good_matches)
# )

# src_pts = np.float32([
#     kp1[i].pt
#     for i, j in good_matches
# ]).reshape(-1, 1, 2)

# dst_pts = np.float32([
#     kp2[j].pt
#     for i, j in good_matches
# ]).reshape(-1, 1, 2)

# H, mask = cv2.findHomography(
#     src_pts,
#     dst_pts,
#     cv2.RANSAC,
#     5.0
# )

# inliers = []

# if mask is not None:

#     mask = mask.ravel()

#     for k in range(len(mask)):

#         if mask[k] == 1:

#             inliers.append(
#                 good_matches[k]
#             )

# print("RANSAC ")

# print(
#     "RANSAC Inliers:",
#     len(inliers)
# )

# result = draw_matches(
#     img1,
#     img2,
#     kp1,
#     kp2,
#     inliers
# )

# cv2.imshow(
#     "Final Matching Result",
#     result
# )

# cv2.waitKey(0)
# cv2.destroyAllWindows()


# demo = gr.Interface(
#     fn=feature_matching_pipeline,
#     inputs=[
#         gr.Image(label="Image 1"),
#         gr.Image(label="Image 2")
#     ],
#     outputs=[
#         gr.Image(label="Final Matching Visualization"),
#         gr.Textbox(label="Statistics")
#     ],
#     title="Feature Matching Mini Project",
#     description="""
# ORB Feature Detection →
# Descriptor Generation →
# Manual KNN Matching →
# Ratio Test →
# RANSAC →
# Final Matching Visualization
# """
# )

# demo.launch()
import cv2
import numpy as np
import gradio as gr


def hamming_distance(desc1, desc2):

    distance = 0

    for a, b in zip(desc1, desc2):

        xor = int(a) ^ int(b)

        while xor:
            distance += xor & 1
            xor >>= 1

    return distance


def knn_match(des1, des2):

    matches = []

    for i in range(len(des1)):

        distances = []

        for j in range(len(des2)):

            d = hamming_distance(
                des1[i],
                des2[j]
            )

            distances.append((j, d))

        distances.sort(
            key=lambda x: x[1]
        )

        matches.append(
            (i, distances[:2])
        )

    return matches


def ratio_test(matches, ratio=0.75):

    good = []

    for query_idx, pair in matches:

        if len(pair) < 2:
            continue

        best = pair[0]
        second = pair[1]

        if best[1] < ratio * second[1]:

            good.append(
                (query_idx, best[0])
            )

    return good


def draw_matches(
        img1,
        img2,
        kp1,
        kp2,
        matches):

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

    for idx1, idx2 in matches:

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


def feature_matching_pipeline(img1, img2):

    if img1 is None or img2 is None:
        return None, "Please upload both images."

    gray1 = cv2.cvtColor(
        img1,
        cv2.COLOR_RGB2GRAY
    )

    gray2 = cv2.cvtColor(
        img2,
        cv2.COLOR_RGB2GRAY
    )

    orb = cv2.ORB_create(
        nfeatures=1000
    )

    kp1, des1 = orb.detectAndCompute(
        gray1,
        None
    )

    kp2, des2 = orb.detectAndCompute(
        gray2,
        None
    )

    if des1 is None or des2 is None:
        return None, "Descriptors could not be generated."

    knn_matches = knn_match(
        des1,
        des2
    )

    total_matches = len(knn_matches)

    good_matches = ratio_test(
        knn_matches,
        ratio=0.75
    )

    if len(good_matches) < 4:
        return None, "Not enough matches for RANSAC."

    src_pts = np.float32([
        kp1[i].pt
        for i, j in good_matches
    ]).reshape(-1, 1, 2)

    dst_pts = np.float32([
        kp2[j].pt
        for i, j in good_matches
    ]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        5.0
    )

    inliers = []

    if mask is not None:

        mask = mask.ravel()

        for k in range(len(mask)):

            if mask[k] == 1:
                inliers.append(
                    good_matches[k]
                )

    result = draw_matches(
        img1,
        img2,
        kp1,
        kp2,
        inliers
    )

    stats = f"""
Image 1 Keypoints: {len(kp1)}
Image 2 Keypoints: {len(kp2)}

Total Matches: {total_matches}

Good Matches: {len(good_matches)}

RANSAC Inliers: {len(inliers)}
"""

    return result, stats


demo = gr.Interface(
    fn=feature_matching_pipeline,
    inputs=[
        gr.Image(label="Image 1"),
        gr.Image(label="Image 2")
    ],
    outputs=[
        gr.Image(label="Final Matching Visualization"),
        gr.Textbox(label="Statistics")
    ],
    title="Feature Matching Mini Project",
    description="""
ORB Feature Detection →
Descriptor Generation →
Manual KNN Matching →
Ratio Test →
RANSAC →
Final Matching Visualization
"""
)

demo.launch()