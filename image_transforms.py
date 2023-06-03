import cv2
import numpy as np

# Trackbar callback function
def update_threshold(value):
    global threshold
    threshold = value / 100.0  # Convert trackbar value to a decimal between 0 and 1

    # Apply thresholding
    thresholded_image = np.where(image > threshold, 1, 0).astype(np.uint8) * 255

    # Display the thresholded image
    cv2.imshow(window_name, thresholded_image)


# Read the infrared image
image_path = "C:/Users/ChengLi-win7/Downloads/archit expts/new_data/azure_kinect1_2_calib_snap/infrared0003.png"
image = cv2.imread(image_path, -1)  # Read as grayscale

image = image.astype(np.float32)
image = (image - np.min(image)) / (np.max(image) - np.min(image))

# Set the initial threshold value
initial_threshold = 10  # Example value (adjust as needed)
threshold = initial_threshold / 100.0  # Convert initial threshold to a decimal between 0 and 1

# Create a window for displaying the image
window_name = "Image Viewer"
cv2.namedWindow(window_name)

# Create a trackbar
trackbar_name = "Threshold"
max_value = 100
cv2.createTrackbar(trackbar_name, window_name, initial_threshold, max_value, update_threshold)

# Apply thresholding
thresholded_image = np.where(image > threshold, 1, 0).astype(np.uint8) * 255

# Display the thresholded image
cv2.imshow(window_name, thresholded_image)

# Wait for key press
key = cv2.waitKey(0) & 0xFF
if key == ord('q'):  # Press 'q' to quit
    exit()

cv2.destroyAllWindows()


# # Preprocessing steps
# # 1. Histogram Equalization
# equalized_image = cv2.equalizeHist(image)

# # 2. Adaptive Thresholding
# adaptive_threshold = cv2.adaptiveThreshold(equalized_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# # 3. Image Filtering (Gaussian Smoothing)
# filtered_image = cv2.GaussianBlur(adaptive_threshold, (5, 5), 0)

# # 4. Morphological Operations (Dilation)
# kernel = np.ones((3, 3), np.uint8)
# dilated_image = cv2.dilate(filtered_image, kernel, iterations=10)

# # 5. Gradient-based Edge Detection (Canny Edge Detection)
# edges = cv2.Canny(image, 50, 51, apertureSize=7, L2gradient=True)

# # Display the results
# cv2.imshow('Original Image', image)
# cv2.imshow('Equalized Image', equalized_image)
# cv2.imshow('Adaptive Threshold', adaptive_threshold)
# cv2.imshow('Filtered Image', filtered_image)
# cv2.imshow('Dilated Image', dilated_image)
# cv2.imshow('Edges', edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
