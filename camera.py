import cv2
import time
from config import CONFIG

class Camera:
    def __init__(self, camera_index=None, window_name=None):
        self.camera_index = camera_index if camera_index is not None else CONFIG["camera"]["index"]
        self.window_name = window_name if window_name is not None else CONFIG["camera"]["window_name"]
        self.cap = None
        self.is_running = False
        self.exit_key = CONFIG["camera"]["exit_key"]
        self.resolution = CONFIG["camera"]["resolution"]
        self.fps = CONFIG["camera"]["fps"]

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera (index: {self.camera_index})")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution["height"])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        self.is_running = True
        print(f"Camera (index: {self.camera_index}) started successfully")

    def stop(self):
        self.is_running = False
        
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        
        cv2.destroyWindow(self.window_name)
        print("Camera stopped and resources released")

    def get_frame(self):
        if not self.is_running or self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        
        if not ret:
            return None
        
        return frame

    def show_frame(self, frame):
        if not self.is_running or frame is None:
            return
        
        cv2.imshow(self.window_name, frame)
        
        from config import CONFIG
        exit_key = CONFIG["camera"]["exit_key"]
        if cv2.waitKey(1) & 0xFF == ord(exit_key):
            self.is_running = False