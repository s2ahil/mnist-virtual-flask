import cv2
import numpy as np

def preprocess(image_path):
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(gray_image, (28, 28))
    inverted_image = cv2.bitwise_not(resized_image)
    return inverted_image
