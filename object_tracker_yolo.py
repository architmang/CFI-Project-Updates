import cv2
import os
import numpy as np

# Load YOLOv8 model and classes
net = cv2.dnn.readNet("yolov3/yolov3.weights", "yolov3/yolov3.cfg")
classes = []
with open("yolov3/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set up object tracker
tracker = cv2.legacy.TrackerCSRT_create()

# Directory containing image frames
data_dir = './AzureKinectRecord_30_05/1/azure_kinect1_2/color'

# Initialize variables
frame_count = 0
tracked_objects = {}

# Process each frame in the directory
for filename in sorted(os.listdir(data_dir)):
    frame_count += 1

    # Read frame
    frame = cv2.imread(os.path.join(data_dir, filename))

    # Object detection using YOLOv8
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    print(layer_names)
    print(cv2.__version__)
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    # output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    # Initialize variables for tracking
    boxes = []
    confidences = []
    class_ids = []

    # Process detection outputs
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'person':
                # Calculate object bounding box
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                # Store detected person details
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform object tracking
    for i in range(len(boxes)):
        if class_ids[i] == 0:  # Check if the detected object is a person
            box = boxes[i]
            left, top, width, height = box

            # Initialize tracker for new person
            if box not in tracked_objects.values():
                tracker = cv2.legacy.TrackerCSRT_create()
                tracker.init(frame, tuple(box))
                tracked_objects[i] = box

            # Update existing trackers
            else:
                success, new_box = tracker.update(frame)
                if success:
                    tracked_objects[i] = new_box

            # Draw bounding box
            cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)

    # Display output
    cv2.imshow("Person Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()
