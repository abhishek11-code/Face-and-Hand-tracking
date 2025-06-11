
## Face and Finger Detection using OpenCV and MediaPipe

This is a real-time face detection and finger counting tool built using Python, OpenCV, and MediaPipe. It captures webcam input and detects:
- Number of faces in the frame
- Number of fingers raised (per hand)

### Features
- Real-time webcam-based detection
- Accurate hand landmark tracking
- Face detection using MediaPipe's pre-trained models
- Safe exit using `q` key or close button

### How to Run
```bash
pip install opencv-python mediapipe
python main.py
