from detection_controller import DetectionController

def show_copyright():
    print("=== Visual Detection Program ===")
    print("Copyright (C) Akira Amatsume\n")

def check_dependencies():
    try:
        import cv2
        import mediapipe
        print("Dependencies check passed")
        return True
    except ImportError as e:
        print(f"Dependencies missing: {e}")
        print("Install required packages: pip install opencv-python mediapipe")
        return False

if __name__ == "__main__":
    if not check_dependencies():
        exit(1)
    show_copyright()
    controller = DetectionController()
    controller.run()