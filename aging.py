import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def apply_wrinkles(img, landmarks, intensity=0.5):
    h, w, _ = img.shape
    wrinkle_layer = np.zeros_like(img, dtype=np.uint8)

    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(wrinkle_layer, (x, y), 1, (255,255,255), -1)

    wrinkle_layer = cv2.GaussianBlur(wrinkle_layer, (15,15), 0)
    return cv2.addWeighted(img, 1, wrinkle_layer, intensity, 0)


def age_effect(image, age_factor=0.7):
    img = image.copy()

    # Desaturate (older skin tone)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = hsv[:,:,1] * (1 - 0.5 * age_factor)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Face mesh detection
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                img = apply_wrinkles(img, face_landmarks.landmark, age_factor)

    # Add noise
    noise = np.random.normal(0, 10 + 20*age_factor, img.shape).astype(np.int16)
    img = img.astype(np.int16) + noise
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Slight blur
    img = cv2.GaussianBlur(img, (3,3), 0)

    return img