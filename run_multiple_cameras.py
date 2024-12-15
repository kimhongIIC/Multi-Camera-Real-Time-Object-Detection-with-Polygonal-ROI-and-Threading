import cv2
import numpy as np
from ultralytics import YOLO
from threading import Thread

# Load the YOLO model
model = YOLO('Model/fall-detection.pt')

# Define class names
class_names = ["falling", "sitting", "standing", "walking"]

# Define the polygon zone for the IP camera (advanced mode)
polygon_zone = np.array([[[8, 7], [29, 446], [282, 444], [277, 3]]])

def is_inside_polygon(polygon, bbox):
    """
    Check if the bounding box center or corners are inside the polygon.
    polygon: np.array of shape (1, N, 2)
    bbox: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    points_to_check = [
        (x1, y1), (x2, y1), (x1, y2), (x2, y2),
        ((x1 + x2) // 2, (y1 + y2) // 2)  # center
    ]
    for (px, py) in points_to_check:
        if cv2.pointPolygonTest(polygon, (px, py), False) >= 0:
            return True
    return False

def detect_fall(frame):
    """
    Perform fall detection and annotate the frame for advanced camera feed.
    Uses the global 'model', 'polygon_zone', and 'class_names'.
    """
    results = model(frame)
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            score = box.conf.item()

            if score < 0.5:  # Skip low-confidence detections
                continue

            bbox = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, bbox)

            # Check if inside the polygon zone
            if is_inside_polygon(polygon_zone, (x1, y1, x2, y2)):
                class_name = "Sleeping" if class_id == 0 else class_names[class_id]
                # Different color if inside polygon
                color = (255, 0, 255) if class_id == 0 else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} ({score:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            else:
                # Default behavior for outside polygon
                color = (0, 0, 255) if class_id == 0 else (0, 255, 0)
                class_name = class_names[class_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} ({score:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame

def advanced_detection_task(source, window_name, window_size=(640, 480)):
    """
    Runs advanced fall detection on the IP camera with polygon zones.
    """
    video = cv2.VideoCapture(source)
    if not video.isOpened():
        print(f"Error: Unable to open video source {source}")
        return

    print(f"Started advanced detection on source: {source}")
    while True:
        ret, frame = video.read()
        if not ret:
            print(f"Stream ended or failed for source: {source}")
            break

        # Resize and detect falls
        frame_resized = cv2.resize(frame, window_size)
        annotated_frame = detect_fall(frame_resized)

        # Draw the polygon zone
        cv2.polylines(annotated_frame, [polygon_zone], isClosed=True, color=(51, 255, 51), thickness=2)

        # Display the annotated frame
        cv2.imshow(window_name, annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyWindow(window_name)

def normal_detection_task(source, window_name, window_size=(640, 480)):
    """
    Runs normal YOLO detection on a video source (like a webcam) without polygon checks.
    """
    model_webcam = YOLO('Model/fall-detection.pt')
    video = cv2.VideoCapture(source)
    if not video.isOpened():
        print(f"Error: Unable to open video source {source}")
        return

    print(f"Started normal detection on source: {source}")
    while True:
        ret, frame = video.read()
        if not ret:
            print(f"Stream ended or failed for source: {source}")
            break

        # Resize frame
        frame_resized = cv2.resize(frame, window_size)

        # Run YOLO prediction
        results = model_webcam.predict(frame_resized, verbose=False)
        # Draw detections
        for box in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = box[:6]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            label = f"{results[0].names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow(window_name, frame_resized)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyWindow(window_name)

# Define video sources and corresponding window names
# Example: An IP camera (advanced detection) and a local webcam (normal detection)
sources = [
    ("rtsp://username:password@IP_camera_address:rstp_port/path", "IP Camera"),
    (0, "Webcam")
]

# Start threads for each video source
threads = []
for source, window_name in sources:
    if isinstance(source, str):
        # Advanced detection for IP camera (string source implies RTSP URL)
        thread = Thread(target=advanced_detection_task, args=(source, window_name))
    else:
        # Normal detection for webcam (integer source)
        thread = Thread(target=normal_detection_task, args=(source, window_name))
    thread.start()
    threads.append(thread)

# Wait for all threads to finish
for thread in threads:
    thread.join()

print("All tasks completed.")
