import cv2
import os
import numpy as np

def display_image(image_path):
    window_name = "Image Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    if os.path.isfile(image_path):
        image = cv2.imread(image_path, -1)
        image = image.astype(np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        # Create a trackbar for dilation
        dilation_value = 1
        max_dilation = 10

        def update_dilation(value):
            nonlocal dilation_value
            dilation_value = value

        cv2.createTrackbar("Dilation", window_name, dilation_value, max_dilation, update_dilation)

        while True:
            # Apply dilation to the image
            kernel_size = 2 * dilation_value + 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            dilated_image = cv2.dilate(image, kernel, iterations=1)

            # Display the image
            cv2.imshow(window_name, dilated_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Press 'q' to quit
                break

    else:
        print(f"Skipping {image_path} as it is not a file.")

    cv2.destroyAllWindows()

img = "C:/Users/ChengLi-win7/Downloads/archit expts/new_data/azure_kinect1_2_calib_snap/infrared0003.png"
display_image(img)

# def display_images(directory):
#     images = sorted(os.listdir(directory))
#     current_index = 0

#     window_name = "Image Viewer"
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#     cv2.resizeWindow(window_name, 800, 600)

#     while current_index < len(images):
#         image_path = os.path.join(directory, images[current_index])

#         if os.path.isfile(image_path):
#             image = cv2.imread(image_path, -1)
#             image = image.astype(np.float32)
#             image = (image - np.min(image)) / (np.max(image) - np.min(image))
#             cv2.imshow(window_name, image)

#             key = cv2.waitKey(0) & 0xFF
#             if key == ord('q'):  # Press 'q' to quit
#                 break
#             elif key == ord('n'):  # Press 'n' to go to the next image
#                 current_index += 1
#         else:
#             print(f"Skipping {image_path} as it is not a file.")

#     cv2.destroyAllWindows()