import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import os

class FaceDetection:
    def __init__(self):
        self.enabled = False
        self.detector = None
        self._init_detector()

    def _init_detector(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "Model", "blaze_face_short_range.tflite")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=0.5
            )
            
            self.detector = vision.FaceDetector.create_from_options(options)
        except Exception as e:
            print(f"Failed to initialize Face Detector: {e}")
            self.detector = None

    def enable(self):
        if self.detector is None:
            print("Face Detection cannot be enabled (model missing/failed to load)")
            return
        self.enabled = True
        print("Face Detection Enabled")

    def disable(self):
        self.enabled = False
        print("Face Detection Disabled")

    def detect(self, frame):
        if not self.enabled or frame is None or self.detector is None:
            return None
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        detection_result = self.detector.detect(mp_image)
        return detection_result

    def draw(self, frame):
        if not self.enabled or frame is None or self.detector is None:
            return frame
        
        frame = cv2.flip(frame, 1)
        detection_result = self.detect(frame)
        
        if not detection_result or not detection_result.detections:
            return frame
        
        frame_height, frame_width = frame.shape[:2]
        
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            x1 = int(bbox.origin_x)
            y1 = int(bbox.origin_y)
            x2 = int(bbox.origin_x + bbox.width)
            y2 = int(bbox.origin_y + bbox.height)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            if detection.categories and len(detection.categories) > 0:
                confidence = detection.categories[0].score
            else:
                confidence = 0.0
            
            cv2.putText(
                frame,
                f"Face: {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )
        
        return frame