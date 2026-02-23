import cv2
import time

class Camera:
    def __init__(self, camera_index=0, window_name="Camera Window"):
        self.camera_index = camera_index
        self.window_name = window_name
        self.cap = None
        self.is_running = False

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera (index: {self.camera_index})")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
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
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.is_running = False