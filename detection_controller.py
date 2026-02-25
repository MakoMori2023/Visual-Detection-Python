import time
import threading
from camera import Camera
from face_detection import FaceDetection
from hand_gesture_detection import HandGestureDetection
from human_stickman_detection import HumanStickmanDetection
from config import CONFIG

class DetectionController:
    def __init__(self):
        self.camera = Camera()

        self.detectors = {
            int(k): (v["name"], eval(v["class"])()) 
            for k, v in CONFIG["detection_controller"]["detectors"].items()
        }
        self.is_running = False
        self.draw_thread = None

        self.draw_fps = CONFIG["detection_controller"]["draw_fps"]
        self.thread_timeout = CONFIG["detection_controller"]["thread_timeout"]
        self.command_prompt = CONFIG["detection_controller"]["commands"]["prompt"]
        self.invalid_cmd_msg = CONFIG["detection_controller"]["commands"]["invalid_msg"]
        self.exit_cmd = CONFIG["detection_controller"]["commands"]["exit_cmd"]

    def show_status(self):
        print("\n--------------------------------------------------")
        print("Current Detection Status:")
        for idx, (name, detector) in self.detectors.items():
            status = "On" if detector.enabled else "Off"
            print(f"   {idx} - {name} -- {status}")
        print("--------------------------------------------------")

    def parse_command(self, cmd):
        cmd = cmd.strip().lower()
        if not cmd:
            return None
        
        if cmd == self.exit_cmd:
            return ("exit", None)
        
        parts = cmd.split()
        if len(parts) != 2:
            return None
        
        action, num_str = parts
        if action not in ["enable", "disable"]:
            return None
        
        try:
            detector_id = int(num_str)
            if detector_id not in self.detectors:
                return None
            return (action, detector_id)
        except ValueError:
            return None

    def toggle_detector(self, action, detector_id):
        name, detector = self.detectors[detector_id]
        if action == "enable":
            detector.enable()
        elif action == "disable":
            detector.disable()

    def _draw_loop(self):
        while self.is_running and self.camera.is_running:
            try:
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                frame = flip_frame(frame)
                
                for idx, (name, detector) in self.detectors.items():
                    frame = detector.draw(frame)
                
                self.camera.show_frame(frame)
                
                time.sleep(1/self.draw_fps)
            except Exception as e:
                print(f"\nDraw loop error: {e}")
                time.sleep(0.01)

    def run(self):
        print("=== Multi-Function Visual Detection Program ===")
        detector_ids = [str(k) for k in self.detectors.keys()]
        detector_keys_str = ', '.join(map(str, self.detectors.keys()))
        print(f"Commands:\n  Enable X (X={detector_keys_str})\n  Disable X (X={detector_keys_str})\n  {self.exit_cmd}\n")
    
        print("Starting camera...")
        try:
            self.camera.start()
        except RuntimeError as e:
            print(f"Failed to start camera, exiting: {e}")
            return
    
        self.is_running = True
        self.draw_thread = threading.Thread(target=self._draw_loop, daemon=True)
        self.draw_thread.start()

        while self.is_running:
            self.show_status()
            cmd = input(self.command_prompt).strip()
            parsed = self.parse_command(cmd)
        
            if not parsed:
                print(self.invalid_cmd_msg)
                continue
        
            action, detector_id = parsed
            if action == "exit":
                break
            self.toggle_detector(action, detector_id)

        self.is_running = False
        self.camera.stop()
        if self.draw_thread is not None and self.draw_thread.is_alive():
            self.draw_thread.join(timeout=self.thread_timeout)
        print("\nShutting down...")
        print("All resources released, program exited safely")

if __name__ == "__main__":
    controller = DetectionController()
    controller.run()