import cv2
import numpy as np
import gradio as gr


def generate_panorama(files):

    if len(files) < 2:
        return None

    images = []

    for file in files:

        img = cv2.imread(file.name)

        if img is not None:
            images.append(img)

    if len(images) < 2:
        return None

    panorama = images[0]

    for i in range(1, len(images)):

        img1 = panorama
        img2 = images[i]

        gray1 = cv2.cvtColor(
            img1,
            cv2.COLOR_BGR2GRAY
        )

        gray2 = cv2.cvtColor(
            img2,
            cv2.COLOR_BGR2GRAY
        )


        orb = cv2.ORB_create(2000)

        kp1, des1 = orb.detectAndCompute(
            gray1,
            None
        )

        kp2, des2 = orb.detectAndCompute(
            gray2,
            None
        )

        if des1 is None or des2 is None:
            continue


        bf = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=True
        )

        matches = bf.match(
            des1,
            des2
        )

        matches = sorted(
            matches,
            key=lambda x: x.distance
        )

        if len(matches) < 4:
            continue

        good_matches = matches[:100]

        src_pts = np.float32(
            [
                kp1[m.queryIdx].pt
                for m in good_matches
            ]
        ).reshape(-1, 1, 2)

        dst_pts = np.float32(
            [
                kp2[m.trainIdx].pt
                for m in good_matches
            ]
        ).reshape(-1, 1, 2)


        H, mask = cv2.findHomography(
            dst_pts,
            src_pts,
            cv2.RANSAC,
            5.0
        )

        if H is None:
            continue

        height = max(
            img1.shape[0],
            img2.shape[0]
        )

        width = (
            img1.shape[1]
            + img2.shape[1]
        )

        

        warped = cv2.warpPerspective(
            img2,
            H,
            (width, height)
        )


        panorama_canvas = np.zeros(
            (
                height,
                width,
                3
            ),
            dtype=np.uint8
        )

        panorama_canvas[
            0:img1.shape[0],
            0:img1.shape[1]
        ] = img1

        # Simple 

        for y in range(height):

            for x in range(width):

                pixel2 = warped[y, x]

                if np.any(pixel2 > 0):

                    pixel1 = panorama_canvas[y, x]

                    if np.any(pixel1 > 0):

                        blended = (
                            pixel1.astype(np.float32)
                            +
                            pixel2.astype(np.float32)
                        ) / 2

                        panorama_canvas[y, x] = blended.astype(
                            np.uint8
                        )

                    else:

                        panorama_canvas[y, x] = pixel2

        panorama = panorama_canvas

    panorama_rgb = cv2.cvtColor(
        panorama,
        cv2.COLOR_BGR2RGB
    )

    return panorama_rgb


interface = gr.Interface(
    fn=generate_panorama,
    inputs=gr.File(
        file_count="multiple",
        label="Upload Overlapping Images"
    ),
    outputs=gr.Image(
        label="Generated Panorama"
    ),
    title="Task 7 - Panorama Generation",
    description="Upload 2 or more overlapping images"
)

interface.launch()