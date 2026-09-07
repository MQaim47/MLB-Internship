import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "images")
LABEL_DIR = os.path.join(BASE_DIR, "dataset", "labels")

classes = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

for root, _, image_names in os.walk(IMAGE_DIR):
    for image_name in image_names:

        if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(root, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        height, width = image.shape[:2]

        relative_path = os.path.relpath(image_path, IMAGE_DIR)
        label_path = os.path.join(
            LABEL_DIR,
            os.path.splitext(relative_path)[0] + ".txt"
        )

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as file:

            lines = file.readlines()

            for line in lines:

                values = line.strip().split()

                class_id = int(values[0])

                x_center = float(values[1])
                y_center = float(values[2])

                box_width = float(values[3])
                box_height = float(values[4])

                x_center *= width
                y_center *= height

                box_width *= width
                box_height *= height

                xmin = int(
                    x_center - box_width / 2
                )

                ymin = int(
                    y_center - box_height / 2
                )

                xmax = int(
                    x_center + box_width / 2
                )

                ymax = int(
                    y_center + box_height / 2
                )

                cv2.rectangle(
                    image,
                    (xmin, ymin),
                    (xmax, ymax),
                    (0, 255, 0),
                    2
                )

                label = classes[class_id]

                cv2.putText(
                    image,
                    label,
                    (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        cv2.imshow(
            image_name,
            image
        )

        cv2.waitKey(0)

cv2.destroyAllWindows()