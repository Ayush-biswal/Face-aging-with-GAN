import cv2
import numpy as np


def add_wrinkles(img, intensity):
    h, w, _ = img.shape

    wrinkle_layer = np.zeros((h, w), dtype=np.uint8)

    for y in range(100, h - 100, 40):
        cv2.line(
            wrinkle_layer,
            (80, y),
            (w - 80, y + np.random.randint(-10, 10)),
            255,
            1
        )

    wrinkle_layer = cv2.GaussianBlur(
        wrinkle_layer,
        (9, 9),
        0
    )

    wrinkle_layer = cv2.cvtColor(
        wrinkle_layer,
        cv2.COLOR_GRAY2BGR
    )

    img = cv2.addWeighted(
        img,
        1,
        wrinkle_layer,
        intensity * 0.15,
        0
    )

    return img


def add_skin_texture(img, intensity):

    noise = np.random.normal(
        0,
        10 + intensity * 15,
        img.shape
    ).astype(np.int16)

    img = img.astype(np.int16) + noise

    img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def desaturate_skin(img, intensity):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv[:, :, 1] = hsv[:, :, 1] * (1 - 0.5 * intensity)

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img


def add_eye_bags(img, intensity):

    h, w, _ = img.shape

    overlay = img.copy()

    cv2.ellipse(
        overlay,
        (w // 2 - 70, h // 2),
        (35, 12),
        0,
        0,
        360,
        (50, 50, 50),
        -1
    )

    cv2.ellipse(
        overlay,
        (w // 2 + 70, h // 2),
        (35, 12),
        0,
        0,
        360,
        (50, 50, 50),
        -1
    )

    img = cv2.addWeighted(
        overlay,
        intensity * 0.25,
        img,
        1 - intensity * 0.25,
        0
    )

    return img


def add_gray_hair(img, intensity):

    h, w, _ = img.shape

    overlay = img.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (w, int(h * 0.22)),
        (180, 180, 180),
        -1
    )

    img = cv2.addWeighted(
        overlay,
        intensity * 0.15,
        img,
        1 - intensity * 0.15,
        0
    )

    return img


def age_effect(image, age_factor=0.6):

    img = image.copy()

    img = desaturate_skin(img, age_factor)

    img = add_wrinkles(img, age_factor)

    img = add_eye_bags(img, age_factor)

    img = add_skin_texture(img, age_factor)

    img = add_gray_hair(img, age_factor)

    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img