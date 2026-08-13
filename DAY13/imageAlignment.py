import cv2
import numpy as np



img1 = cv2.imread("image.jpg")
img2 = cv2.imread("image2.jpg")

if img1 is None or img2 is None:
    print("Images not found!")
    exit()



gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)



orb = cv2.ORB_create(1000)

kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)



bf = cv2.BFMatcher(cv2.NORM_HAMMING)

matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)



good_matches = matches[:100]



src_pts = []
dst_pts = []

for match in good_matches:

    src_pts.append(
        kp1[match.queryIdx].pt
    )

    dst_pts.append(
        kp2[match.trainIdx].pt
    )

src_pts = np.float32(src_pts)
dst_pts = np.float32(dst_pts)



H, mask = cv2.findHomography(
    dst_pts,
    src_pts,
    cv2.RANSAC,
    5.0
)

print("Homography Matrix:\n")
print(H)



height, width = img1.shape[:2]

aligned = cv2.warpPerspective(
    img2,
    H,
    (width, height)
)



cv2.imshow("Image 1", img1)
cv2.imshow("Image 2", img2)
cv2.imshow("Aligned Image 2", aligned)

cv2.waitKey(0)
cv2.destroyAllWindows()