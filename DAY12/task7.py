import pandas as pd

detectors = [
    ["Harris", "Corner Detector", "Medium", "High", "No", "No"],
    ["Shi-Tomasi", "Corner Detector", "Medium", "Very High", "No", "No"],
    ["FAST", "Keypoint Detector", "Very Fast", "Medium", "No", "No"],
    ["DoG", "Blob Detector", "Slow", "High", "Yes", "Yes"]
]

descriptors = [
    ["SIFT", "128", "Float", "Slow", "Yes", "Yes"],
    ["ORB", "32", "Binary", "Fast", "Partial", "Yes"],
    ["BRIEF", "32", "Binary", "Very Fast", "No", "No"],
    ["BRISK", "64", "Binary", "Fast", "Yes", "Yes"]
]

matching = [
    ["BF Matcher", "High", "Slow"],
    ["KNN Matcher", "Very High", "Medium"],
    ["Ratio Test", "Best", "Fast"]
]

detector_table = pd.DataFrame(
    detectors,
    columns=[
        "Detector",
        "Type",
        "Speed",
        "Accuracy",
        "Scale Invariant",
        "Rotation Invariant"
    ]
)

descriptor_table = pd.DataFrame(
    descriptors,
    columns=[
        "Descriptor",
        "Size",
        "Type",
        "Speed",
        "Scale Invariant",
        "Rotation Invariant"
    ]
)

matching_table = pd.DataFrame(
    matching,
    columns=[
        "Matching Method",
        "Accuracy",
        "Speed"
    ]
)

print("\nDETECTOR COMPARISON\n")
print(detector_table)

print("\nDESCRIPTOR COMPARISON\n")
print(descriptor_table)

print("\nMATCHING COMPARISON\n")
print(matching_table)