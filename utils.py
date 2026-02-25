import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from typing import List, Tuple, Optional, Any

def load_mediapipe_model(
    model_path: str,
    task_type: str,
    **kwargs
) -> Optional[Any]:
    if not os.path.exists(model_path):
        print(f"[Utils Error] Model isn't exists: {model_path}")
        return None

    try:
        base_options = python.BaseOptions(model_asset_path=model_path)
        
        if task_type == "face_detector":
            min_detection_confidence = kwargs.get("min_detection_confidence", 0.5)
            options = vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=min_detection_confidence
            )
            detector = vision.FaceDetector.create_from_options(options)
        
        elif task_type == "hand_landmarker":
            num_hands = kwargs.get("num_hands", 2)
            min_detection_conf = kwargs.get("min_hand_detection_confidence", 0.5)
            min_presence_conf = kwargs.get("min_hand_presence_confidence", 0.5)
            min_tracking_conf = kwargs.get("min_tracking_confidence", 0.5)
            
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=num_hands,
                min_hand_detection_confidence=min_detection_conf,
                min_hand_presence_confidence=min_presence_conf,
                min_tracking_confidence=min_tracking_conf
            )
            detector = vision.HandLandmarker.create_from_options(options)
        
        elif task_type == "pose_landmarker":
            num_poses = kwargs.get("num_poses", 1)
            min_detection_conf = kwargs.get("min_pose_detection_confidence", 0.5)
            min_presence_conf = kwargs.get("min_pose_presence_confidence", 0.5)
            min_tracking_conf = kwargs.get("min_tracking_confidence", 0.5)
            
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                num_poses=num_poses,
                min_pose_detection_confidence=min_detection_conf,
                min_pose_presence_confidence=min_presence_conf,
                min_tracking_confidence=min_tracking_conf,
                output_segmentation_masks=False
            )
            detector = vision.PoseLandmarker.create_from_options(options)
        
        else:
            print(f"[Utils Error] Unsupported task types: {task_type}")
            return None
        
        print(f"[Utils Info] {task_type} Loading success")
        return detector
    
    except Exception as e:
        print(f"[Utils Error] Loading fail: {str(e)}")
        return None

def convert_bgr_to_mp_image(frame: cv2.Mat) -> Optional[mp.Image]:
    if frame is None:
        return None
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    except Exception as e:
        print(f"[Utils Error] Image format conversion recognition failed: {str(e)}")
        return None

def flip_frame(frame: cv2.Mat) -> cv2.Mat:
    if frame is None:
        return frame
    return cv2.flip(frame, 1)

def draw_landmark_points(
    frame: cv2.Mat,
    landmarks: List[Any],
    frame_size: Tuple[int, int],
    circle_radius: int = 5,
    circle_color: Tuple[int, int, int] = (255, 0, 0)
) -> cv2.Mat:
    if frame is None or not landmarks:
        return frame
    
    frame_height, frame_width = frame_size
    for landmark in landmarks:
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        cv2.circle(frame, (x, y), circle_radius, circle_color, -1)
    return frame

def draw_landmark_connections(
    frame: cv2.Mat,
    landmarks: List[Any],
    connections: List[Tuple[int, int]],
    frame_size: Tuple[int, int],
    line_thickness: int = 2,
    line_color: Tuple[int, int, int] = (0, 255, 0)
) -> cv2.Mat:
    if frame is None or not landmarks or not connections:
        return frame
    
    frame_height, frame_width = frame_size
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue
        
        start_landmark = landmarks[start_idx]
        end_landmark = landmarks[end_idx]
        
        start_x = int(start_landmark.x * frame_width)
        start_y = int(start_landmark.y * frame_height)
        end_x = int(end_landmark.x * frame_width)
        end_y = int(end_landmark.y * frame_height)
        
        cv2.line(frame, (start_x, start_y), (end_x, end_y), line_color, line_thickness)
    return frame

def draw_bounding_box(
    frame: cv2.Mat,
    bbox: Any,
    box_color: Tuple[int, int, int] = (0, 0, 255),
    box_thickness: int = 2,
    label: Optional[str] = None,
    label_config: Optional[dict] = None
) -> cv2.Mat:

    if frame is None or bbox is None:
        return frame
    
    x1 = int(bbox.origin_x)
    y1 = int(bbox.origin_y)
    x2 = int(bbox.origin_x + bbox.width)
    y2 = int(bbox.origin_y + bbox.height)
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)
    
    if label and label_config:
        cv2.putText(
            frame,
            label,
            (x1, y1 + label_config["offset_y"]),
            label_config["font"],
            label_config["scale"],
            tuple(label_config["color"]),
            label_config["thickness"]
        )
    return frame