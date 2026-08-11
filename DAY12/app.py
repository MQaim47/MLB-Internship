# import gradio as gr
# import numpy as np

# from task1 import task1_keypoints
# from task2 import task2_corners
# from task3 import task3_fast
# from task4 import task4_descriptors
# from task5 import task5_distance
# from task6 import task6_matching
# from task7 import task7_comparison
# from miniProject import feature_matching_pipeline


# def task1_tab(image):

#     image = np.array(image)

#     result, report = task1_keypoints(
#         image,
#         0.25
#     )

#     return result, report


# def task2_tab(image):

#     image = np.array(image)

#     harris, shi, report = task2_corners(
#         image,
#         0.01
#     )

#     return harris, shi, report


# def task3_tab(image):

#     image = np.array(image)

#     result, report = task3_fast(
#         image,
#         20
#     )

#     return result, report


# def task4_tab(image):

#     image = np.array(image)

#     result, report = task4_descriptors(
#         image
#     )

#     return result, report


# def task5_tab():

#     return task5_distance()


# def task6_tab(image1, image2):

#     image1 = np.array(image1)
#     image2 = np.array(image2)

#     result, report = task6_matching(
#         image1,
#         image2
#     )

#     return result, report


# def task7_tab():

#     return task7_comparison()


# def mini_project_tab(image1, image2):

#     image1 = np.array(image1)
#     image2 = np.array(image2)

#     result, stats = feature_matching_pipeline(
#         image1,
#         image2
#     )

#     return result, stats


# with gr.Blocks(
#     title="Module 12 - Feature Detection"
# ) as demo:

#     gr.Markdown(
#         "# Module 12 - Feature Detection, Description & Matching"
#     )

#     with gr.Tab("Task 1 - Keypoints"):

#         inp = gr.Image(type="numpy")

#         btn = gr.Button("Run")

#         out_img = gr.Image(
#             label="Detected Keypoints"
#         )

#         out_txt = gr.Textbox(
#             label="Observation"
#         )

#         btn.click(
#             task1_tab,
#             inp,
#             [out_img, out_txt]
#         )

#     with gr.Tab("Task 2 - Harris & Shi-Tomasi"):

#         inp2 = gr.Image(type="numpy")

#         btn2 = gr.Button("Run")

#         harris = gr.Image(
#             label="Harris"
#         )

#         shi = gr.Image(
#             label="Shi-Tomasi"
#         )

#         report2 = gr.Textbox()

#         btn2.click(
#             task2_tab,
#             inp2,
#             [harris, shi, report2]
#         )

#     with gr.Tab("Task 3 - FAST"):

#         inp3 = gr.Image(type="numpy")

#         btn3 = gr.Button("Run")

#         fast_img = gr.Image()

#         fast_report = gr.Textbox()

#         btn3.click(
#             task3_tab,
#             inp3,
#             [fast_img, fast_report]
#         )

#     with gr.Tab("Task 4 - Descriptors"):

#         inp4 = gr.Image(type="numpy")

#         btn4 = gr.Button("Run")

#         desc_img = gr.Image()

#         desc_report = gr.Textbox()

#         btn4.click(
#             task4_tab,
#             inp4,
#             [desc_img, desc_report]
#         )

#     with gr.Tab("Task 5 - Distance Metrics"):

#         btn5 = gr.Button(
#             "Calculate"
#         )

#         report5 = gr.Textbox(
#             lines=20
#         )

#         btn5.click(
#             task5_tab,
#             None,
#             report5
#         )

#     with gr.Tab("Task 6 - Feature Matching"):

#         img1 = gr.Image(type="numpy")

#         img2 = gr.Image(type="numpy")

#         btn6 = gr.Button("Match")

#         match_img = gr.Image()

#         match_report = gr.Textbox()

#         btn6.click(
#             task6_tab,
#             [img1, img2],
#             [match_img, match_report]
#         )

#     with gr.Tab("Task 7 - Comparison"):

#         btn7 = gr.Button(
#             "Show Comparison"
#         )

#         report7 = gr.Textbox(
#             lines=25
#         )

#         btn7.click(
#             task7_tab,
#             None,
#             report7
#         )

#     with gr.Tab("Mini Project"):

#         pimg1 = gr.Image(type="numpy")

#         pimg2 = gr.Image(type="numpy")

#         pbtn = gr.Button(
#             "Run Pipeline"
#         )

#         pout = gr.Image()

#         pstats = gr.Textbox()

#         pbtn.click(
#             mini_project_tab,
#             [pimg1, pimg2],
#             [pout, pstats]
#         )

# demo.launch()

"""
Day 12 - Feature Detection, Description & Matching
====================================================
Single Gradio app combining Task 1-7 and the Mini Project into one
tabbed interface.

Run with:
    pip install -r requirements.txt
    python app.py

A local URL (and a public gradio.live link if share=True) will be printed
in the terminal.
"""

import os
import time

import numpy as np
import pandas as pd
import cv2
import gradio as gr

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
IMG_PATH = os.path.join(ASSET_DIR, "image.jpg")
OBJ1_PATH = os.path.join(ASSET_DIR, "object1.jpg")
OBJ2_PATH = os.path.join(ASSET_DIR, "object2.jpg")


# ============================================================
# Shared helpers
# ============================================================

def to_gray(image_rgb):
    """Gradio hands us RGB uint8 arrays. Grayscale conversion is safe
    regardless of channel order (only used for intensity from here on)."""
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


def resize_max_dim(image, max_dim):
    """Down-scale so the longer side is at most max_dim.

    Tasks 1-3 use hand-written pure-Python pixel loops (no vectorization),
    so on a full-size photo they can take a long time. Shrinking the image
    keeps the demo responsive while leaving the algorithm untouched.
    """
    h, w = image.shape[:2]
    scale = max_dim / float(max(h, w))
    if scale >= 1.0:
        return image
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def require_image(image):
    if image is None:
        raise gr.Error("Please upload an image first.")
    return image


def require_two_images(img1, img2):
    if img1 is None or img2 is None:
        raise gr.Error("Please upload both images.")
    return img1, img2


def default_image(path):
    return path if os.path.exists(path) else None


# ============================================================
# Shared: manual convolution + Sobel gradients (Task 1 & Task 2)
# ============================================================

def convolution(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    output = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            region = padded[y:y + kh, x:x + kw]
            output[y, x] = np.sum(region * kernel)
    return output


def calculate_gradients(gray):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = convolution(gray, sobel_x)
    gy = convolution(gray, sobel_y)
    return gx, gy


# ============================================================
# Task 1 - Gradient-magnitude keypoint detector
# ============================================================

def detect_keypoints_t1(gray, threshold_ratio=0.25):
    gx, gy = calculate_gradients(gray)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    threshold = threshold_ratio * magnitude.max()
    keypoints = magnitude > threshold
    return keypoints, magnitude


def draw_keypoints_t1(image, keypoints, color=(255, 0, 0)):
    output = image.copy()
    ys, xs = np.where(keypoints)
    for y, x in zip(ys, xs):
        cv2.circle(output, (int(x), int(y)), 2, color, -1)
    return output


def run_task1(image, threshold_ratio, max_dim):
    image = require_image(image)
    image = resize_max_dim(image, int(max_dim))
    gray = to_gray(image).astype(np.float32)
    keypoints, _ = detect_keypoints_t1(gray, threshold_ratio)
    output = draw_keypoints_t1(image, keypoints)
    info = (f"Total keypoints: {int(np.sum(keypoints))}\n"
            f"Processed at {image.shape[1]}x{image.shape[0]} "
            f"(resized for speed - manual convolution is pure Python)")
    return output, info


# ============================================================
# Task 2 - Harris vs Shi-Tomasi corner detectors
# ============================================================

def box_filter(image, size=3):
    kernel = np.ones((size, size), dtype=np.float32) / (size * size)
    return convolution(image, kernel)


def harris_detector(gray, k=0.04, threshold_ratio=0.01):
    Ix, Iy = calculate_gradients(gray)
    Ix2, Iy2, Ixy = Ix * Ix, Iy * Iy, Ix * Iy
    Sx2 = box_filter(Ix2, 3)
    Sy2 = box_filter(Iy2, 3)
    Sxy = box_filter(Ixy, 3)
    det = (Sx2 * Sy2) - (Sxy * Sxy)
    trace = Sx2 + Sy2
    R = det - k * (trace ** 2)
    threshold = threshold_ratio * R.max()
    return R > threshold, R


def shi_tomasi_detector(gray, threshold_ratio=0.01):
    Ix, Iy = calculate_gradients(gray)
    Ix2, Iy2, Ixy = Ix * Ix, Iy * Iy, Ix * Iy
    Sx2 = box_filter(Ix2, 3)
    Sy2 = box_filter(Iy2, 3)
    Sxy = box_filter(Ixy, 3)
    trace = Sx2 + Sy2
    det = (Sx2 * Sy2) - (Sxy * Sxy)
    temp = np.sqrt(np.maximum(trace * trace - 4 * det, 0))
    lambda1 = (trace + temp) / 2
    lambda2 = (trace - temp) / 2
    R = np.minimum(lambda1, lambda2)
    threshold = threshold_ratio * R.max()
    return R > threshold, R


def draw_corners(image, corners, color):
    output = image.copy()
    ys, xs = np.where(corners)
    for y, x in zip(ys, xs):
        cv2.circle(output, (int(x), int(y)), 2, color, -1)
    return output


def run_task2(image, k, threshold_ratio, max_dim):
    image = require_image(image)
    image = resize_max_dim(image, int(max_dim))
    gray = to_gray(image).astype(np.float32)

    harris_corners, _ = harris_detector(gray, k=k, threshold_ratio=threshold_ratio)
    shi_corners, _ = shi_tomasi_detector(gray, threshold_ratio=threshold_ratio)

    harris_output = draw_corners(image, harris_corners, (255, 0, 0))
    shi_output = draw_corners(image, shi_corners, (0, 255, 0))

    info = (f"Harris corners: {int(np.sum(harris_corners))}\n"
            f"Shi-Tomasi corners: {int(np.sum(shi_corners))}\n"
            f"Processed at {image.shape[1]}x{image.shape[0]}")
    return harris_output, shi_output, info


# ============================================================
# Task 3 - Manual FAST-style corner detector
# ============================================================

_FAST_CIRCLE = [
    (0, -3), (1, -3), (2, -2), (3, -1), (3, 0), (3, 1), (2, 2), (1, 3),
    (0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3),
]


def get_circle_pixels(image, x, y):
    return [image[y + dy, x + dx] for dx, dy in _FAST_CIRCLE]


def check_fast_corner(circle_values, center, threshold, n=12):
    bright = [v > center + threshold for v in circle_values]
    dark = [v < center - threshold for v in circle_values]
    bright = bright + bright
    dark = dark + dark

    count = 0
    for v in bright:
        count = count + 1 if v else 0
        if count >= n:
            return True

    count = 0
    for v in dark:
        count = count + 1 if v else 0
        if count >= n:
            return True

    return False


def fast_detector(gray, threshold=20):
    # Cast to a signed, wider type first: gray is uint8, and
    # "center + threshold" / "center - threshold" can silently wrap
    # around at the 0/255 boundary in uint8 arithmetic otherwise.
    gray = gray.astype(np.int16)
    h, w = gray.shape
    keypoints = []
    for y in range(3, h - 3):
        for x in range(3, w - 3):
            center = gray[y, x]
            circle_values = get_circle_pixels(gray, x, y)
            if check_fast_corner(circle_values, center, threshold):
                keypoints.append((x, y))
    return keypoints


def draw_keypoints_t3(image, keypoints, color=(255, 0, 0)):
    output = image.copy()
    for x, y in keypoints:
        cv2.circle(output, (x, y), 2, color, -1)
    return output


def run_task3_single(image, threshold, max_dim):
    image = require_image(image)
    image = resize_max_dim(image, int(max_dim))
    gray = to_gray(image)
    keypoints = fast_detector(gray, int(threshold))
    output = draw_keypoints_t3(image, keypoints)
    info = (f"Threshold {int(threshold)} -> {len(keypoints)} keypoints\n"
            f"Processed at {image.shape[1]}x{image.shape[0]}")
    return output, info


def run_task3_compare(image, max_dim):
    image = require_image(image)
    image = resize_max_dim(image, int(max_dim))
    gray = to_gray(image)
    gallery = []
    lines = []
    for threshold in [10, 20, 30, 50]:
        keypoints = fast_detector(gray, threshold)
        output = draw_keypoints_t3(image, keypoints)
        gallery.append((output, f"Threshold {threshold} ({len(keypoints)} pts)"))
        lines.append(f"Threshold {threshold}: {len(keypoints)} keypoints")
    return gallery, "\n".join(lines)


# ============================================================
# Task 4 - SIFT vs ORB vs BRISK vs BRIEF descriptor comparison
# ============================================================

def _create_brisk():
    # BRISK's location has moved between OpenCV builds/versions - try the
    # common spots instead of hard-crashing the whole comparison.
    if hasattr(cv2, "BRISK_create"):
        return cv2.BRISK_create()
    if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "BRISK_create"):
        return cv2.xfeatures2d.BRISK_create()
    raise AttributeError("BRISK is not available in this OpenCV build")


def run_task4(image):
    image = require_image(image)
    gray = to_gray(image)

    results = []
    gallery = []

    detectors = [
        ("SIFT", cv2.SIFT_create, "Float"),
        ("ORB", cv2.ORB_create, "Binary"),
        ("BRISK", _create_brisk, "Binary"),
    ]

    for name, factory, dtype in detectors:
        try:
            detector = factory()
            start = time.time()
            kp, des = detector.detectAndCompute(gray, None)
            end = time.time()
            results.append([name, len(kp), des.shape[1] if des is not None else 0, dtype, round(end - start, 5)])
            gallery.append((cv2.drawKeypoints(image, kp, None, color=(0, 255, 0)), name))
        except Exception:
            results.append([name, 0, 0, dtype, "unavailable in this OpenCV build"])

    try:
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        fast = cv2.FastFeatureDetector_create()
        kp = fast.detect(gray, None)
        start = time.time()
        kp, des = brief.compute(gray, kp)
        end = time.time()
        results.append(["BRIEF", len(kp), des.shape[1] if des is not None else 0, "Binary", round(end - start, 5)])
        gallery.append((cv2.drawKeypoints(image, kp, None, color=(0, 255, 0)), "BRIEF"))
    except Exception:
        results.append(["BRIEF", 0, 0, "Binary", "opencv-contrib required"])

    table = pd.DataFrame(results, columns=["Descriptor", "Keypoints", "Descriptor Size", "Type", "Time (s)"])
    return table, gallery


# ============================================================
# Task 5 - Descriptor distance calculator (Euclidean & Hamming)
# ============================================================

def euclidean_distance(d1, d2):
    total = 0
    for a, b in zip(d1, d2):
        total += (a - b) ** 2
    return total ** 0.5


def hamming_distance_bits(d1, d2):
    return sum(1 for a, b in zip(d1, d2) if a != b)


def _parse_numbers(text):
    return [float(v.strip()) for v in text.split(",") if v.strip() != ""]


def _parse_bits(text):
    values = [v.strip() for v in text.replace(",", " ").split() if v.strip() != ""]
    bits = [int(v) for v in values]
    for b in bits:
        if b not in (0, 1):
            raise gr.Error("Binary descriptor values must be 0 or 1.")
    return bits


def run_task5_euclidean(d1_text, d2_text):
    d1, d2 = _parse_numbers(d1_text), _parse_numbers(d2_text)
    if len(d1) != len(d2):
        raise gr.Error("Both descriptors must have the same length.")
    if len(d1) == 0:
        raise gr.Error("Enter comma-separated numbers, e.g. 12, 24, 36")
    dist = euclidean_distance(np.array(d1), np.array(d2))
    return f"Euclidean distance = {round(dist, 4)}"


def run_task5_hamming(d1_text, d2_text):
    d1, d2 = _parse_bits(d1_text), _parse_bits(d2_text)
    if len(d1) != len(d2):
        raise gr.Error("Both binary descriptors must have the same length.")
    if len(d1) == 0:
        raise gr.Error("Enter 0/1 values, e.g. 1 0 1 1 0 1 0 1")
    dist = hamming_distance_bits(d1, d2)
    return f"Hamming distance = {dist}"


def task5_reference_tables():
    euclid_examples = [
        ([1, 2, 3, 4], [2, 3, 4, 5]),
        ([10, 20, 30, 40], [15, 25, 35, 45]),
        ([100, 120, 140, 160], [110, 130, 150, 170]),
    ]
    rows = []
    for i, (a, b) in enumerate(euclid_examples, start=1):
        dist = euclidean_distance(np.array(a), np.array(b))
        rows.append([f"Example {i}", str(a), str(b), round(dist, 4)])
    euclid_df = pd.DataFrame(rows, columns=["#", "Descriptor 1", "Descriptor 2", "Euclidean Distance"])

    binary_examples = [
        ([1, 0, 1, 0, 1, 0, 1, 0], [1, 1, 1, 0, 0, 0, 1, 1]),
        ([1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 1, 0, 1, 0, 1]),
        ([0, 0, 0, 0, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0, 1, 0]),
    ]
    rows = []
    for i, (a, b) in enumerate(binary_examples, start=1):
        dist = hamming_distance_bits(a, b)
        rows.append([f"Example {i}", str(a), str(b), dist])
    hamming_df = pd.DataFrame(rows, columns=["#", "Descriptor 1", "Descriptor 2", "Hamming Distance"])
    return euclid_df, hamming_df


# ============================================================
# Task 6 - ORB matching: brute-force vs KNN + ratio test
# ============================================================

def hamming_distance_bytes(desc1, desc2):
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
            distance = hamming_distance_bytes(descriptors1[i], descriptors2[j])
            if distance < best_distance:
                best_distance = distance
                best_index = j
        matches.append((i, best_index, best_distance))
    return matches


def knn_match_t6(descriptors1, descriptors2, k=2):
    all_matches = []
    for i in range(len(descriptors1)):
        distances = []
        for j in range(len(descriptors2)):
            distance = hamming_distance_bytes(descriptors1[i], descriptors2[j])
            distances.append((j, distance))
        distances.sort(key=lambda x: x[1])
        all_matches.append(distances[:k])
    return all_matches


def ratio_test_t6(knn_matches, ratio=0.75):
    # The original task6.py lost the query index i here (it appended just
    # the (j, distance) pair), which fed the wrong values into
    # draw_matches as if they were keypoint indices. enumerate() keeps
    # (query_idx, match_idx, distance) so the visualization lines up.
    good_matches = []
    for i, match_pair in enumerate(knn_matches):
        if len(match_pair) < 2:
            continue
        best, second = match_pair[0], match_pair[1]
        if best[1] < ratio * second[1]:
            good_matches.append((i, best[0], best[1]))
    return good_matches


def draw_matches_generic(img1, img2, kp1, kp2, matches):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2
    for match in matches:
        idx1, idx2 = match[0], match[1]
        x1, y1 = kp1[idx1].pt
        x2, y2 = kp2[idx2].pt
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2) + w1, int(y2)
        cv2.circle(canvas, (x1, y1), 4, (0, 255, 0), -1)
        cv2.circle(canvas, (x2, y2), 4, (0, 255, 0), -1)
        cv2.line(canvas, (x1, y1), (x2, y2), (255, 0, 0), 1)
    return canvas


def run_task6(img1, img2, n_features, ratio):
    img1, img2 = require_two_images(img1, img2)
    gray1, gray2 = to_gray(img1), to_gray(img2)

    orb = cv2.ORB_create(nfeatures=int(n_features))
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        raise gr.Error("Could not find descriptors in one of the images. Try a different image.")

    bf_matches = brute_force_match(des1, des2)
    bf_matches.sort(key=lambda x: x[2])
    bf_result = draw_matches_generic(img1, img2, kp1, kp2, bf_matches[:50])

    knn_matches = knn_match_t6(des1, des2, k=2)
    good_matches = ratio_test_t6(knn_matches, ratio=ratio)
    ratio_result = draw_matches_generic(img1, img2, kp1, kp2, good_matches[:50])

    info = (f"Image 1 keypoints: {len(kp1)} | Image 2 keypoints: {len(kp2)}\n"
            f"Brute-force matches: {len(bf_matches)} (top 50 shown)\n"
            f"Good matches after ratio test: {len(good_matches)} (top 50 shown)")
    return bf_result, ratio_result, info


# ============================================================
# Task 7 - Reference comparison tables
# ============================================================

def build_task7_tables():
    detectors = [
        ["Harris", "Corner Detector", "Medium", "High", "No", "No"],
        ["Shi-Tomasi", "Corner Detector", "Medium", "Very High", "No", "No"],
        ["FAST", "Keypoint Detector", "Very Fast", "Medium", "No", "No"],
        ["DoG", "Blob Detector", "Slow", "High", "Yes", "Yes"],
    ]
    descriptors = [
        ["SIFT", "128", "Float", "Slow", "Yes", "Yes"],
        ["ORB", "32", "Binary", "Fast", "Partial", "Yes"],
        ["BRIEF", "32", "Binary", "Very Fast", "No", "No"],
        ["BRISK", "64", "Binary", "Fast", "Yes", "Yes"],
    ]
    matching = [
        ["BF Matcher", "High", "Slow"],
        ["KNN Matcher", "Very High", "Medium"],
        ["Ratio Test", "Best", "Fast"],
    ]
    detector_table = pd.DataFrame(detectors, columns=["Detector", "Type", "Speed", "Accuracy", "Scale Invariant", "Rotation Invariant"])
    descriptor_table = pd.DataFrame(descriptors, columns=["Descriptor", "Size", "Type", "Speed", "Scale Invariant", "Rotation Invariant"])
    matching_table = pd.DataFrame(matching, columns=["Matching Method", "Accuracy", "Speed"])
    return detector_table, descriptor_table, matching_table


# ============================================================
# Mini Project - ORB + manual KNN + Ratio Test + RANSAC
# ============================================================

def knn_match_mp(des1, des2):
    matches = []
    for i in range(len(des1)):
        distances = []
        for j in range(len(des2)):
            d = hamming_distance_bytes(des1[i], des2[j])
            distances.append((j, d))
        distances.sort(key=lambda x: x[1])
        matches.append((i, distances[:2]))
    return matches


def ratio_test_mp(matches, ratio=0.75):
    good = []
    for query_idx, pair in matches:
        if len(pair) < 2:
            continue
        best, second = pair[0], pair[1]
        if best[1] < ratio * second[1]:
            good.append((query_idx, best[0]))
    return good


def feature_matching_pipeline(img1, img2, n_features, ratio, ransac_thresh):
    img1, img2 = require_two_images(img1, img2)
    gray1, gray2 = to_gray(img1), to_gray(img2)

    orb = cv2.ORB_create(nfeatures=int(n_features))
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        raise gr.Error("Descriptors could not be generated for one of the images.")

    knn_matches = knn_match_mp(des1, des2)
    total_matches = len(knn_matches)
    good_matches = ratio_test_mp(knn_matches, ratio=ratio)

    if len(good_matches) < 4:
        raise gr.Error(
            f"Only {len(good_matches)} good matches found - need at least 4 for RANSAC. "
            f"Try a lower ratio or a higher feature count."
        )

    src_pts = np.float32([kp1[i].pt for i, j in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[j].pt for i, j in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, float(ransac_thresh))

    inliers = []
    if mask is not None:
        mask = mask.ravel()
        for k in range(len(mask)):
            if mask[k] == 1:
                inliers.append(good_matches[k])

    result = draw_matches_generic(img1, img2, kp1, kp2, inliers)

    stats = (f"Image 1 keypoints: {len(kp1)}\n"
             f"Image 2 keypoints: {len(kp2)}\n\n"
             f"Total KNN matches: {total_matches}\n"
             f"Good matches (ratio test): {len(good_matches)}\n"
             f"RANSAC inliers: {len(inliers)}")
    return result, stats


# ============================================================
# Gradio UI
# ============================================================

with gr.Blocks(title="Day 12 - Feature Detection & Matching") as demo:
    gr.Markdown(
        """
        # Day 12 - Feature Detection, Description & Matching
        Manual (from-scratch) implementations of corner/keypoint detectors, descriptor
        comparisons, distance metrics, and feature matching. Each task is its own tab.
        """
    )

    with gr.Tabs():
        # ---------------- Task 1 ----------------
        with gr.Tab("Task 1 - Gradient Keypoints"):
            gr.Markdown("Custom Sobel-gradient-magnitude keypoint detector (manual convolution).")
            with gr.Row():
                with gr.Column():
                    t1_image = gr.Image(value=default_image(IMG_PATH), type="numpy", label="Input Image")
                    t1_threshold = gr.Slider(0.05, 0.9, value=0.25, step=0.01, label="Threshold Ratio")
                    t1_maxdim = gr.Slider(80, 400, value=250, step=10, label="Resize (max dimension, for speed)")
                    t1_button = gr.Button("Detect Keypoints", variant="primary")
                with gr.Column():
                    t1_output = gr.Image(label="Keypoints")
                    t1_info = gr.Textbox(label="Info", lines=3)
            t1_button.click(run_task1, inputs=[t1_image, t1_threshold, t1_maxdim], outputs=[t1_output, t1_info])

        # ---------------- Task 2 ----------------
        with gr.Tab("Task 2 - Harris vs Shi-Tomasi"):
            gr.Markdown("Manual Harris and Shi-Tomasi corner detectors built from the same structure-tensor math.")
            with gr.Row():
                with gr.Column():
                    t2_image = gr.Image(value=default_image(IMG_PATH), type="numpy", label="Input Image")
                    t2_k = gr.Slider(0.01, 0.10, value=0.04, step=0.005, label="Harris k")
                    t2_threshold = gr.Slider(0.001, 0.10, value=0.01, step=0.001, label="Threshold Ratio")
                    t2_maxdim = gr.Slider(80, 400, value=250, step=10, label="Resize (max dimension, for speed)")
                    t2_button = gr.Button("Detect Corners", variant="primary")
                with gr.Column():
                    t2_harris_out = gr.Image(label="Harris Corners")
                    t2_shi_out = gr.Image(label="Shi-Tomasi Corners")
                    t2_info = gr.Textbox(label="Info", lines=3)
            t2_button.click(run_task2, inputs=[t2_image, t2_k, t2_threshold, t2_maxdim],
                             outputs=[t2_harris_out, t2_shi_out, t2_info])

        # ---------------- Task 3 ----------------
        with gr.Tab("Task 3 - FAST Detector"):
            gr.Markdown(
                "Manual FAST-16 corner detector (pure Python pixel loop) - this is the slowest task, "
                "so the image is auto-downscaled before processing."
            )
            with gr.Row():
                with gr.Column():
                    t3_image = gr.Image(value=default_image(IMG_PATH), type="numpy", label="Input Image")
                    t3_maxdim = gr.Slider(60, 250, value=150, step=10, label="Resize (max dimension, for speed)")
                    t3_threshold = gr.Slider(5, 80, value=20, step=1, label="Threshold")
                    t3_button = gr.Button("Detect Corners", variant="primary")
                    t3_compare_button = gr.Button("Compare Thresholds 10 / 20 / 30 / 50 (slower)")
                with gr.Column():
                    t3_output = gr.Image(label="Keypoints")
                    t3_gallery = gr.Gallery(label="Threshold Comparison", columns=2)
                    t3_info = gr.Textbox(label="Info", lines=4)
            t3_button.click(run_task3_single, inputs=[t3_image, t3_threshold, t3_maxdim],
                             outputs=[t3_output, t3_info])
            t3_compare_button.click(run_task3_compare, inputs=[t3_image, t3_maxdim],
                                     outputs=[t3_gallery, t3_info])

        # ---------------- Task 4 ----------------
        with gr.Tab("Task 4 - Descriptor Comparison"):
            gr.Markdown("SIFT vs ORB vs BRISK vs BRIEF: keypoint count, descriptor size, type, and compute time.")
            with gr.Row():
                with gr.Column():
                    t4_image = gr.Image(value=default_image(IMG_PATH), type="numpy", label="Input Image")
                    t4_button = gr.Button("Compare Descriptors", variant="primary")
                with gr.Column():
                    t4_table = gr.Dataframe(label="Comparison Table")
            t4_gallery = gr.Gallery(label="Keypoints per Method", columns=4)
            t4_button.click(run_task4, inputs=[t4_image], outputs=[t4_table, t4_gallery])

        # ---------------- Task 5 ----------------
        with gr.Tab("Task 5 - Descriptor Distances"):
            gr.Markdown(
                "Euclidean distance (for float descriptors like SIFT) and Hamming distance "
                "(for binary descriptors like ORB/BRIEF)."
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Euclidean distance**")
                    t5_e1 = gr.Textbox(value="12, 24, 36, 48, 60, 72, 84, 96, 108, 120",
                                        label="Descriptor 1 (comma-separated numbers)")
                    t5_e2 = gr.Textbox(value="10, 20, 30, 50, 65, 70, 80, 100, 110, 125",
                                        label="Descriptor 2 (comma-separated numbers)")
                    t5_e_button = gr.Button("Compute Euclidean Distance")
                    t5_e_result = gr.Textbox(label="Result")
                with gr.Column():
                    gr.Markdown("**Hamming distance**")
                    t5_h1 = gr.Textbox(value="1 0 1 1 0 1 0 1 1 1 0 0 1 0 1 1",
                                        label="Descriptor 1 (0/1 values)")
                    t5_h2 = gr.Textbox(value="1 1 1 0 0 1 1 1 1 0 0 1 1 0 1 0",
                                        label="Descriptor 2 (0/1 values)")
                    t5_h_button = gr.Button("Compute Hamming Distance")
                    t5_h_result = gr.Textbox(label="Result")
            t5_e_button.click(run_task5_euclidean, inputs=[t5_e1, t5_e2], outputs=[t5_e_result])
            t5_h_button.click(run_task5_hamming, inputs=[t5_h1, t5_h2], outputs=[t5_h_result])

            gr.Markdown("**Reference examples (from the original script)**")
            _euclid_df, _hamming_df = task5_reference_tables()
            gr.Dataframe(value=_euclid_df, label="Euclidean examples")
            gr.Dataframe(value=_hamming_df, label="Hamming examples")

        # ---------------- Task 6 ----------------
        with gr.Tab("Task 6 - ORB Matching"):
            gr.Markdown("ORB features matched two ways: brute-force nearest neighbour vs manual KNN + Lowe's ratio test.")
            with gr.Row():
                with gr.Column():
                    t6_img1 = gr.Image(value=default_image(OBJ1_PATH), type="numpy", label="Image 1")
                    t6_img2 = gr.Image(value=default_image(OBJ2_PATH), type="numpy", label="Image 2")
                    t6_nfeatures = gr.Slider(50, 500, value=150, step=10,
                                              label="ORB feature count (higher = slower, matching is manual Python)")
                    t6_ratio = gr.Slider(0.5, 0.95, value=0.75, step=0.01, label="Ratio test threshold")
                    t6_button = gr.Button("Match Features", variant="primary")
                with gr.Column():
                    t6_bf_out = gr.Image(label="Brute-Force Matches (top 50)")
                    t6_ratio_out = gr.Image(label="KNN + Ratio Test Matches (top 50)")
                    t6_info = gr.Textbox(label="Info", lines=3)
            t6_button.click(run_task6, inputs=[t6_img1, t6_img2, t6_nfeatures, t6_ratio],
                             outputs=[t6_bf_out, t6_ratio_out, t6_info])

        # ---------------- Task 7 ----------------
        with gr.Tab("Task 7 - Reference Tables"):
            gr.Markdown("Quick-reference comparison tables for detectors, descriptors, and matching strategies.")
            _det_df, _desc_df, _match_df = build_task7_tables()
            gr.Dataframe(value=_det_df, label="Detector Comparison")
            gr.Dataframe(value=_desc_df, label="Descriptor Comparison")
            gr.Dataframe(value=_match_df, label="Matching Comparison")

        # ---------------- Mini Project ----------------
        with gr.Tab("Mini Project - Full Pipeline"):
            gr.Markdown("ORB detection -> manual KNN matching -> ratio test -> RANSAC homography -> final visualization.")
            with gr.Row():
                with gr.Column():
                    mp_img1 = gr.Image(value=default_image(OBJ1_PATH), type="numpy", label="Image 1")
                    mp_img2 = gr.Image(value=default_image(OBJ2_PATH), type="numpy", label="Image 2")
                    mp_nfeatures = gr.Slider(50, 500, value=300, step=10, label="ORB feature count (higher = slower)")
                    mp_ratio = gr.Slider(0.5, 0.95, value=0.85, step=0.01, label="Ratio test threshold")
                    mp_ransac = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="RANSAC reprojection threshold")
                    mp_button = gr.Button("Run Pipeline", variant="primary")
                with gr.Column():
                    mp_output = gr.Image(label="Final Matching Visualization")
                    mp_stats = gr.Textbox(label="Statistics", lines=6)
            mp_button.click(feature_matching_pipeline, inputs=[mp_img1, mp_img2, mp_nfeatures, mp_ratio, mp_ransac],
                             outputs=[mp_output, mp_stats])


if __name__ == "__main__":
    demo.launch()