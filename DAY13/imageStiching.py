import cv2
import numpy as np


def stitch_images(img1, aligned):

    h1, w1 = img1.shape[:2]
    h2, w2 = aligned.shape[:2]

    panorama_height = max(h1, h2)
    panorama_width = max(w1, w2)

    panorama = np.zeros(
        (
            panorama_height,
            panorama_width,
            3
        ),
        dtype=np.uint8
    )



    panorama[0:h1, 0:w1] = img1



    for y in range(h2):

        for x in range(w2):

            pixel2 = aligned[y, x]

            if np.any(pixel2 > 0):

                pixel1 = panorama[y, x]


                if np.any(pixel1 > 0):

                    blended = (
                        pixel1.astype(np.float32)
                        +
                        pixel2.astype(np.float32)
                    ) / 2

                    panorama[y, x] = blended.astype(np.uint8)

                else:

                    panorama[y, x] = pixel2

    return panorama



img1 = cv2.imread("image.jpg")

aligned = cv2.imread("image2.jpg")

if img1 is None:
    print("image.jpg not found")
    exit()

if aligned is None:
    print("aligned.jpg not found")
    exit()



panorama = stitch_images(
    img1,
    aligned
)




cv2.imwrite(
    "panorama.jpg",
    panorama
)



cv2.imshow(
    "Image 1",
    img1
)

cv2.imshow(
    "Aligned Image",
    aligned
)

cv2.imshow(
    "Panorama",
    panorama
)

cv2.waitKey(0)
cv2.destroyAllWindows()