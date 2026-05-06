import cv2
import numpy as np


def age_effect(image, age_factor=0.7):
    img = image.copy()

    # Desaturate skin tone
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * (1 - 0.5 * age_factor)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Add realistic noise
    noise = np.random.normal(
        0,
        10 + 20 * age_factor,
        img.shape
    ).astype(np.int16)

    img = img.astype(np.int16) + noise
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Slight blur
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Dark circles effect
    h, w, _ = img.shape
    cv2.ellipse(
        img,
        (w // 3, h // 2),
        (30, 10),
        0,
        0,
        360,
        (40, 40, 40),
        -1
    )

    cv2.ellipse(
        img,
        (2 * w // 3, h // 2),
        (30, 10),
        0,
        0,
        360,
        (40, 40, 40),
        -1
    )

    return img