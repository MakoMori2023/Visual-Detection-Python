import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

class HandGestureDetection:
    def __init__(self):
        self.enabled = False
        self.detector = None
        self._init_detector()
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]

    def _init_detector(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "Model", "hand_landmarker.task")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Hand model file not found: {model_path}")
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,

                num_hands=2,

                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.detector = vision.HandLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Failed to initialize Hand Gesture Detector: {e}")
            self.detector = None

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
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        detection_result = self.detector.detect(mp_image)
        return detection_result

    def draw(self, frame):
        if not self.enabled or frame is None or self.detector is None:
            return frame
        
        frame = cv2.flip(frame, 1)
        detection_result = self.detect(frame)
        
        if not detection_result or not detection_result.hand_landmarks:
            return frame
        
        frame_height, frame_width = frame.shape[:2]
        
        for hand_landmarks in detection_result.hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)


                # cv2.putText(frame, str(idx), (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            
            for connection in self.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start_landmark = hand_landmarks[start_idx]
                end_landmark = hand_landmarks[end_idx]
                
                start_x = int(start_landmark.x * frame_width)
                start_y = int(start_landmark.y * frame_height)
                end_x = int(end_landmark.x * frame_width)
                end_y = int(end_landmark.y * frame_height)
                
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        
        return frame