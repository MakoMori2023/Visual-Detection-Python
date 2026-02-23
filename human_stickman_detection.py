import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

class HumanStickmanDetection:
    def __init__(self):
        self.enabled = False
        self.detector = None
        self._init_detector()
        self.POSE_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8), 
            (9, 10),
            (11, 12),
            (11, 13), (13, 15), (15, 17), (17, 19), (19, 21),
            (12, 14), (14, 16), (16, 18), (18, 20), (20, 22),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (27, 29), (29, 31),
            (24, 26), (26, 28), (28, 30), (30, 32) 
        ]

    def _init_detector(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "Model", "pose_landmarker_full.task")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Pose model not found: {model_path}\n请确认 pose_landmarker_full.task 已放入 Model 文件夹")
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False
            )
            
            self.detector = vision.PoseLandmarker.create_from_options(options)
            print("Human Stickman Detector initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Human Stickman Detector: {str(e)}")
            self.detector = None

    def enable(self):
        if self.detector is None:
            print("Human Stick Figure Detection cannot be enabled (model missing/failed to load)")
            return
        self.enabled = True
        print("Human Stick Figure Detection enabled")

    def disable(self):
        self.enabled = False
        print("Human Stick Figure Detection disabled")

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
        
        if not detection_result or not detection_result.pose_landmarks:
            return frame
        
        frame_height, frame_width = frame.shape[:2]
        
        for pose_landmarks in detection_result.pose_landmarks:
            for idx, landmark in enumerate(pose_landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)


                # cv2.putText(frame, str(idx), (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
            
            for connection in self.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx >= len(pose_landmarks) or end_idx >= len(pose_landmarks):
                    continue
                
                start_landmark = pose_landmarks[start_idx]
                end_landmark = pose_landmarks[end_idx]
                
                start_x = int(start_landmark.x * frame_width)
                start_y = int(start_landmark.y * frame_height)
                end_x = int(end_landmark.x * frame_width)
                end_y = int(end_landmark.y * frame_height)
                
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 3)
        
        return frame