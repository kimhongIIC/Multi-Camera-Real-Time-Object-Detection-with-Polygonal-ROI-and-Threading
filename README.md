# Multi-Camera Real-Time Object Detection with Polygonal ROI and Threading

This project demonstrates how to run YOLO-based object detection on multiple video streams (e.g., an IP camera and a local webcam) simultaneously using Python's threading. One camera feed is processed with an "advanced" detection mode that includes a polygonal region of interest (ROI), while another feed is processed with a "normal" detection mode. This setup is useful for applications like monitoring multiple surveillance feeds or testing different detection logic simultaneously.

**Key Features:**
- **Multiple Camera Feeds:** Process multiple video sources at the same time (e.g., an RTSP IP camera feed and a local webcam).
- **Threading for Concurrency:** Each video source runs in its own thread, ensuring that one slow source does not block the others.
- **Advanced Detection Mode with Polygonal ROI:** One of the camera feeds includes logic to define a polygonal zone. Objects inside this polygon can be highlighted differently than objects outside.
- **Normal Detection Mode:** A simpler detection pipeline for another video source (like a local webcam) without polygonal constraints.

## How It Works

1. **Model Loading:**
   - Uses a YOLO model (`fall-detection.pt`) for object detection.
   - Both the advanced and normal detection modes load the YOLO model and run inference in real-time.

2. **Multiple Sources:**
   - Configure `sources` in the code with either RTSP URLs (for IP cameras) or integer indices (for local webcams).
   - Each source is processed in a separate thread, improving concurrency and responsiveness.

3. **Advanced Detection:**
   - For the IP camera, the code defines a polygonal ROI (`polygon_zone`).
   - Objects inside this polygon can be processed differently (e.g., drawn in a different color or labeled differently).
   - Fall detection logic can be incorporated here, as seen in the example.

4. **Normal Detection:**
   - For a local webcam, normal YOLO detection is performed without additional polygon checks.
   - Objects are simply detected, boxed, and labeled.

## Requirements

- Python 3.x
- OpenCV (`pip install opencv-python`)
- [Ultralytics YOLO](https://docs.ultralytics.com/) (`pip install ultralytics`)
- A `best.pt` YOLO model file
- One or more video sources:
  - RTSP URL for an IP camera
  - Local webcam (use `0` for the default webcam)

## Usage

1. Place `fall-detection.pt` in the project directory.
2. Edit the `sources` list in `run_multiple_cameras.py` to match your camera setup. For example:
   ```python
   sources = [
       ("rtsp://username:password@192.168.x.x:554/stream", "Advanced IP Camera"),
       (0, "Webcam")
   ]
