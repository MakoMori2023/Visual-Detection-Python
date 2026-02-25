import cv2
from config import CONFIG
from utils import (
    load_mediapipe_model,
    convert_bgr_to_mp_image,
    flip_frame,
    draw_landmark_points,
    draw_landmark_connections
)

class HandGestureDetection:
    def __init__(self):
        self.enabled = False
        self.detector = None

        self.model_path = CONFIG["hand_gesture_detection"]["model_path"]
        self.num_hands = CONFIG["hand_gesture_detection"]["num_hands"]
        self.min_hand_detection_confidence = CONFIG["hand_gesture_detection"]["min_hand_detection_confidence"]
        self.min_hand_presence_confidence = CONFIG["hand_gesture_detection"]["min_hand_presence_confidence"]
        self.min_tracking_confidence = CONFIG["hand_gesture_detection"]["min_tracking_confidence"]
        self.HAND_CONNECTIONS = CONFIG["hand_gesture_detection"]["hand_connections"]
        self.draw_config = CONFIG["hand_gesture_detection"]["draw"]

        self._init_detector()

    def _init_detector(self):
        self.detector = load_mediapipe_model(
            model_path=self.model_path,
            task_type="hand_landmarker",
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_hand_detection_confidence,
            min_hand_presence_confidence=self.min_hand_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

    def enable(self):
        if self.detector is None:
            print("Hand Gesture Detection cannot be enabled (model missing/failed to load)")
            return
        self.enabled = True
        print("Hand Gesture Detection enabled")

    def disable(self):
        self.enabled = False
        print("Hand Gesture Detection disabled")

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
        
        if not detection_result or not detection_result.hand_landmarks:
            return frame
        
        frame_size = frame.shape[:2]  # (height, width)
        
        for hand_landmarks in detection_result.hand_landmarks:

            frame = draw_landmark_points(
                frame=frame,
                landmarks=hand_landmarks,
                frame_size=frame_size,
                circle_radius=self.draw_config["circle_radius"],
                circle_color=tuple(self.draw_config["circle_color"])
            )

            frame = draw_landmark_connections(
                frame=frame,
                landmarks=hand_landmarks,
                connections=self.HAND_CONNECTIONS,
                frame_size=frame_size,
                line_thickness=self.draw_config["line_thickness"],
                line_color=tuple(self.draw_config["line_color"])
            )
        
        return frame