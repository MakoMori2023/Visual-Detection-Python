import cv2
from config import CONFIG
from utils import (
    load_mediapipe_model,
    convert_bgr_to_mp_image,
    flip_frame,
    draw_bounding_box
)

class FaceDetection:
    def __init__(self):
        self.enabled = False
        self.detector = None
        self.model_path = CONFIG["face_detection"]["model_path"]
        self.min_detection_confidence = CONFIG["face_detection"]["min_detection_confidence"]
        self.draw_config = CONFIG["face_detection"]["draw"]
        self._init_detector()

    def _init_detector(self):
        self.detector = load_mediapipe_model(
            model_path=self.model_path,
            task_type="face_detector",
            min_detection_confidence=self.min_detection_confidence
        )

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

        mp_image = convert_bgr_to_mp_image(frame)
        if mp_image is None:
            return None
        return self.detector.detect(mp_image)

    def draw(self, frame):
        if not self.enabled or frame is None or self.detector is None:
            return frame
        
        # frame = flip_frame(frame)
        detection_result = self.detect(frame)
        
        if not detection_result or not detection_result.detections:
            return frame
        
        for detection in detection_result.detections:

            confidence = detection.categories[0].score if (detection.categories and len(detection.categories) > 0) else 0.0
            label = f"Face: {confidence:.2f}"
            
            frame = draw_bounding_box(
                frame=frame,
                bbox=detection.bounding_box,
                box_color=tuple(self.draw_config["box_color"]),
                box_thickness=self.draw_config["box_thickness"],
                label=label,
                label_config=self.draw_config["text"]
            )
        
        return frame