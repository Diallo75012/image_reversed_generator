#!/usr/bin/python3

import numpy as np
import os
import cv2
from PIL import Image
from skimage import color, segmentation

# with scikit-image edge detection and color quantization results to create a more detailed coloring image.

def convert_to_coloring_image(img_path):
    image = cv2.imread(img_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection using Sobel operator
    sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.hypot(sobel_x, sobel_y)
    sobel[sobel > 255] = 255
    sobel_uint8 = sobel.astype(np.uint8)

    # Color quantization
    image_lab = color.rgb2lab(image)
    image_quant = segmentation.slic(image_lab, n_segments=32, compactness=30, sigma=1)
    image_quant_rgb = color.label2rgb(image_quant, image, alpha=0.3, kind='avg')

    image_quant_gray = cv2.cvtColor(np.float32(image_quant_rgb), cv2.COLOR_RGB2GRAY)
    _, image_quant_binary = cv2.threshold(image_quant_gray, 127, 255, cv2.THRESH_BINARY)

    # Convert image_quant_binary to the same data type as sobel_uint8
    image_quant_binary_uint8 = image_quant_binary.astype(np.uint8)

    # Combine edge detection result and color quantization result
    combined_image = cv2.addWeighted(sobel_uint8, 0.7, image_quant_binary_uint8, 0.3, 0)

    return combined_image

"""
# delated technique using cv2

def convert_to_coloring_image(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (15, 15), 0)

    # Apply adaptive thresholding
    adaptive_threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 10)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(adaptive_threshold, kernel, iterations=1)

    return dilated
"""

"""
# Reverting technique

def convert_to_coloring_image(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    canny = cv2.Canny(blurred, 50, 200)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(canny, kernel, iterations=1)
    inverted = cv2.bitwise_not(dilated)

    return inverted
"""

def save_coloring_image(image, output_path):
    img_pil = Image.fromarray(image)
    img_pil.save(output_path)

input_folder = '/mnt/c/Users/sibma/Downloads/input_images'
output_folder = '/mnt/c/Users/sibma/Downloads/output_images'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f'coloring_{filename}')
        
        coloring_image = convert_to_coloring_image(input_path)
        save_coloring_image(coloring_image, output_path)

print("Conversion completed.")

