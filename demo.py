import os
import cv2
import mediapipe as mp

def display_images(directory):
    images = sorted(os.listdir(directory))
    current_index = 0

    window_name = "Image Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    while current_index < len(images):
        image_path = os.path.join(directory, images[current_index])

        if os.path.isfile(image_path):
            image = cv2.imread(image_path)
            cv2.imshow(window_name, image)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):  # Press 'q' to quit
                break
            elif key == ord('n'):  # Press 'n' to go to the next image
                current_index += 1
        else:
            print(f"Skipping {image_path} as it is not a file.")

    cv2.destroyAllWindows()

def perform_pose_estimation(input_directory, output_directory):
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Create MediaPipe Holistic object
    mp_holistic = mp.solutions.holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5)

    # Process images
    images = sorted(os.listdir(input_directory))
    for image_name in images:
        image_path = os.path.join(input_directory, image_name)
        output_path = os.path.join(output_directory, image_name)

        if os.path.isfile(image_path):
            image = cv2.imread(image_path)

            # Convert BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Perform holistic pose estimation
            results = mp_holistic.process(image_rgb)

            # Draw poses on the image
            if results.pose_landmarks is not None:
                mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)

            # Save image with pose overlay
            cv2.imwrite(output_path, image)

            print(f"Processed image: {image_path}")

    mp_holistic.close()

# Provide the input and output directory paths
input_directory = "archit_color"
output_directory = "archit_color_new"

# Call the function to perform pose estimation and save new images
perform_pose_estimation(input_directory, output_directory)

directory_path = output_directory
display_images(directory_path)


