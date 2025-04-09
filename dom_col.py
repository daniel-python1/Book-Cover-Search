import os
import cv2
import numpy as np

image_folder = "book_covers"
filename = "0_Alice's_Adventures_in_Wonderland.jpg"

full_path = os.path.normpath(os.path.join(image_folder, filename))
print("Checking file:", full_path)
print("Exists?", os.path.exists(full_path))

image = cv2.imread(full_path)
if image is None:
    print("Failed to read image.")
else:
    print("Image loaded:", image.shape)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((-1, 3))
    image = np.float32(image)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(image, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    color = centers[0].astype(int)
    print(f"Dominant color: rgb({color[0]}, {color[1]}, {color[2]})")
