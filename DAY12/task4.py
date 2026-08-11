import cv2
import time
import pandas as pd

image = cv2.imread("image.jpg")

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

results = []

sift = cv2.SIFT_create()

start = time.time()

kp, des = sift.detectAndCompute(
    gray,
    None
)

end = time.time()

results.append([
    "SIFT",
    len(kp),
    des.shape[1] if des is not None else 0,
    "Float",
    round(end - start, 5)
])

orb = cv2.ORB_create()

start = time.time()

kp, des = orb.detectAndCompute(
    gray,
    None
)

end = time.time()

results.append([
    "ORB",
    len(kp),
    des.shape[1] if des is not None else 0,
    "Binary",
    round(end - start, 5)
])

brisk = cv2.BRISK_create()

start = time.time()

kp, des = brisk.detectAndCompute(
    gray,
    None
)

end = time.time()

results.append([
    "BRISK",
    len(kp),
    des.shape[1] if des is not None else 0,
    "Binary",
    round(end - start, 5)
])

try:

    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    fast = cv2.FastFeatureDetector_create()

    kp = fast.detect(
        gray,
        None
    )

    start = time.time()

    kp, des = brief.compute(
        gray,
        kp
    )

    end = time.time()

    results.append([
        "BRIEF",
        len(kp),
        des.shape[1] if des is not None else 0,
        "Binary",
        round(end - start, 5)
    ])

except:

    results.append([
        "BRIEF",
        0,
        0,
        "Binary",
        "opencv-contrib required"
    ])

table = pd.DataFrame(
    results,
    columns=[
        "Descriptor",
        "Keypoints",
        "Descriptor Size",
        "Type",
        "Time (s)"
    ]
)

print(table)

for descriptor_name in ["SIFT", "ORB", "BRISK"]:

    if descriptor_name == "SIFT":
        detector = cv2.SIFT_create()

    elif descriptor_name == "ORB":
        detector = cv2.ORB_create()

    else:
        detector = cv2.BRISK_create()

    kp, des = detector.detectAndCompute(
        gray,
        None
    )

    output = cv2.drawKeypoints(
        image,
        kp,
        None,
        color=(0, 255, 0)
    )

    cv2.imshow(
        descriptor_name,
        output
    )

cv2.waitKey(0)
cv2.destroyAllWindows()