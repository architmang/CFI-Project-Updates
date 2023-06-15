import os
import cv2
import numpy as np
import torch
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from torchvision.models import resnet50
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from yolov5.models.experimental import attempt_load
from yolov5.utils import torch_utils
from yolov5.utils.general import non_max_suppression

# Load YOLOv5 model and classes
weights = 'yolov5s.pt'
model = attempt_load(weights, device='cpu').float()
model.eval()

# Set device for YOLOv5 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load pre-trained ResNet-50 model
resnet_model = resnet50(pretrained=True)
resnet_model.eval()

# DeepSORT parameters
max_cosine_distance = 0.5  # Maximum cosine distance for data association
nn_budget = None  # Size of the tracking feature embedding dimension
nms_max_overlap = 1.0  # Non-maxima suppression maximum overlap

# Initialize DeepSORT objects
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# Directory containing image frames
data_dir = './AzureKinectRecord_30_05/1/azure_kinect1_2/color'
frame_count = 0

# Process each frame in the directory
for filename in sorted(os.listdir(data_dir)):
    frame_count += 1

    # Read frame
    frame = cv2.imread(os.path.join(data_dir, filename))
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
    # Reshape the tensor to the desired size
    resize_transform = transforms.Resize((1280, 1280))
    resized_tensor = resize_transform(frame_tensor)

    # Perform object detection with YOLOv5
    with torch.no_grad():
        detections = model(resized_tensor)
        conf_threshold = 0.5  # Confidence threshold for non-maximum suppression
        iou_threshold = 0.5  # IoU threshold for non-maximum suppression
        detections = non_max_suppression(detections, conf_threshold, iou_threshold)

    # Convert YOLOv5 results to DeepSORT format
    features = []
    boxes = []
    confidences = []
    for detection in detections:
        if detection is not None and len(detection) > 0:
            class_id = detection[:, -1].int()
            confidences.append(detection[:, 4])
            boxes.append(detection[:, :4])

            # Extract feature embeddings using ResNet-50
            for bbox in detection[:, :4]:
                roi = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

                # Check if roi is not None and has a valid size
                if roi is not None and roi.size != 0:
                    roi = cv2.resize(roi, (224, 224))
                    roi_tensor = TF.to_tensor(roi).unsqueeze(0)
                    features.append(resnet_model(roi_tensor).detach().numpy().flatten())

    if len(boxes) > 0:
        boxes = torch.cat(boxes, dim=0)
        confidences = torch.cat(confidences, dim=0)
        features = np.array(features)
        detections = [Detection(bbox, confidence, feature) for bbox, confidence, feature in zip(boxes, confidences, features)]
    else:
        detections = []

    # Perform tracking with DeepSORT
    tracker.predict()
    tracker.update(detections)

    # Display frame with bounding boxes and track IDs
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr().astype(int)
        track_id = track.track_id
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, str(track_id), (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display frame with bounding boxes and track IDs
    cv2.imshow("Object Tracking", frame)
    cv2.waitKey(1)

