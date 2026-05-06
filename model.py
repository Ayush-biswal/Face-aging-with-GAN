import numpy as np
import cv2

def age_effect(image):
    img = np.array(image)

    # Convert to grayscale and back (aging feel)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Add slight blur (skin aging)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Add noise (wrinkles effect)
    noise = np.random.normal(0, 25, blur.shape).astype(np.uint8)
    aged = cv2.add(blur, noise)

    return aged