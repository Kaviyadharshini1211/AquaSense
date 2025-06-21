import cv2
import numpy as np
import tifffile


def preprocess_image(img):
    # Resize image
    img = cv2.resize(img, (224, 224))

    # Normalize image (if it's not already normalized)
    img = cv2.normalize(img.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    # Optionally apply other preprocessing steps like contrast enhancement, etc.
    return img
