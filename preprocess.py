import cv2
import numpy as np

IMG_SIZE = 128

def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 127.5 - 1
    return image