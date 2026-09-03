detections = [
    ("dog", 0.95),
    ("cat", 0.82),
    ("car", 0.30),
    ("person", 0.65)
]

threshold = 0.50

for label, confidence in detections:
    if confidence >= threshold:
        print(label, confidence)