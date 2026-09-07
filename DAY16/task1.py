import cv2
import os

# VOC -> YOLO Conversion

def voc_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height):

    x_center = ((xmin + xmax) / 2) / img_width
    y_center = ((ymin + ymax) / 2) / img_height

    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height

    return x_center, y_center, width, height


# YOLO -> VOC Conversion

def yolo_to_voc(x_center, y_center, width, height,
                img_width, img_height):

    x_center *= img_width
    y_center *= img_height

    width *= img_width
    height *= img_height

    xmin = int(x_center - width / 2)
    ymin = int(y_center - height / 2)

    xmax = int(x_center + width / 2)
    ymax = int(y_center + height / 2)

    return xmin, ymin, xmax, ymax


# IoU Calculation

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection_area = (
        intersection_width *
        intersection_height
    )

    area_box1 = (
        (box1[2] - box1[0]) *
        (box1[3] - box1[1])
    )

    area_box2 = (
        (box2[2] - box2[0]) *
        (box2[3] - box2[1])
    )

    union_area = (
        area_box1 +
        area_box2 -
        intersection_area
    )

    if union_area == 0:
        return 0

    return intersection_area / union_area


# Test IoU

boxA = [100, 100, 300, 300]
boxB = [150, 150, 350, 350]

iou = calculate_iou(boxA, boxB)

print("\nIoU Example")
print("IoU =", round(iou, 4))


# IoU = 0 Example

boxC = [50, 50, 100, 100]
boxD = [200, 200, 300, 300]

print(
    "IoU = 0 Example:",
    calculate_iou(boxC, boxD)
)

# IoU = 1 Example

boxE = [100, 100, 300, 300]
boxF = [100, 100, 300, 300]

print(
    "IoU = 1 Example:",
    calculate_iou(boxE, boxF)
)



# Process Images

image_folder = "images"

for image_name in os.listdir(image_folder):

    image_path = os.path.join(
        image_folder,
        image_name
    )

    image = cv2.imread(image_path)

    if image is None:
        continue

    h, w = image.shape[:2]

    # Example Bounding Box
    xmin = 50
    ymin = 50
    xmax = 250
    ymax = 250

    # Draw VOC Box
    cv2.rectangle(
        image,
        (xmin, ymin),
        (xmax, ymax),
        (0, 255, 0),
        2
    )

    # Convert to YOLO
    x_center, y_center, bw, bh = voc_to_yolo(
        xmin,
        ymin,
        xmax,
        ymax,
        w,
        h
    )

    print("\nImage:", image_name)

    print("VOC Format:")
    print(
        xmin,
        ymin,
        xmax,
        ymax
    )

    print("YOLO Format:")
    print(
        round(x_center, 4),
        round(y_center, 4),
        round(bw, 4),
        round(bh, 4)
    )

    # Convert Back
    voc_box = yolo_to_voc(
        x_center,
        y_center,
        bw,
        bh,
        w,
        h
    )

    print("Back To VOC:")
    print(voc_box)

    cv2.imshow(
        image_name,
        image
    )

    cv2.waitKey(0)

cv2.destroyAllWindows()