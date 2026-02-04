import sys
import subprocess
import logging
import time
import msvcrt
from threading import Thread, Lock
import math

logging.basicConfig(
    level=logging.ERROR,
    format="Abnormal termination File:%(filename)s | Line:%(lineno)d | Error message:%(message)s",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

def check_missing_dependencies():
    required_deps = {"opencv-python": "cv2", "mediapipe": "mediapipe"}
    missing = []
    for pip_package, import_module in required_deps.items():
        try:
            __import__(import_module)
        except ImportError:
            missing.append(pip_package)
    return missing

def update_installation_progress(progress):
    global INSTALLATION_PROGRESS
    with INSTALLATION_LOCK:
        INSTALLATION_PROGRESS = min(max(progress, 0.0), 1.0)

def draw_installation_progress_bar(progress):
    bar_length = 50
    filled_length = int(bar_length * progress)
    progress_bar = "=" * filled_length + "-" * (bar_length - filled_length)
    print(
        f"\rDependencies installation progress: [{progress_bar}] {progress*100:.1f}%",
        end="",
        flush=True
    )

def get_pip_installation_progress(process):
    update_installation_progress(0.0)
    step_increment = 1.0 / 20
    while process.poll() is None:
        line = process.stdout.readline()
        if line:
            update_installation_progress(INSTALLATION_PROGRESS + step_increment)
            draw_installation_progress_bar(INSTALLATION_PROGRESS)
        time.sleep(0.1)
    update_installation_progress(1.0)
    draw_installation_progress_bar(INSTALLATION_PROGRESS)
    print()

def safe_check_and_install_dependencies():
    missing_deps = check_missing_dependencies()
    cv2_module = None
    mp_module = None
    gesture_detection_enabled = False
    if not missing_deps:
        import cv2
        import mediapipe as mp
        cv2_module = cv2
        mp_module = mp
        gesture_detection_enabled = True
        print("All required dependencies are installed. Gesture detection enabled.")
        return cv2_module, mp_module, gesture_detection_enabled
    print(f"Missing dependencies: {', '.join(missing_deps)}")
    print("Gesture detection requires these dependencies. Install now?")
    print("Press Y to confirm, ESC to cancel (case-sensitive)")
    while True:
        key = capture_keyboard_input()
        if key == "ESC":
            print("\nInstallation cancelled. Only basic camera function (if available) will run.")
            try:
                import cv2
                cv2_module = cv2
                import mediapipe as mp
                mp_module = mp
            except ImportError:
                cv2_module = None
                mp_module = None
            return cv2_module, mp_module, False
        elif key == "Y":
            print(f"\nInstalling dependencies: {', '.join(missing_deps)}...")
            install_cmd = PIP_INSTALL_COMMAND + missing_deps
            process = subprocess.Popen(
                install_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            progress_thread = Thread(
                target=get_pip_installation_progress,
                args=(process,),
                daemon=True
            )
            progress_thread.start()
            try:
                process.wait(timeout=INSTALLATION_TIMEOUT)
                progress_thread.join()
                if process.returncode == 0:
                    remaining_missing = check_missing_dependencies()
                    if not remaining_missing:
                        import cv2
                        import mediapipe as mp
                        cv2_module = cv2
                        mp_module = mp
                        gesture_detection_enabled = True
                        print("All dependencies installed. Gesture detection enabled.")
                    else:
                        print(f"Installation completed but missing: {', '.join(remaining_missing)}")
                else:
                    print(f"Installation failed (return code: {process.returncode})")
            except subprocess.TimeoutExpired:
                process.kill()
                progress_thread.join()
                print(f"Installation timed out (>{INSTALLATION_TIMEOUT}s)")
            return cv2_module, mp_module, gesture_detection_enabled
        elif key is not None:
            print("Invalid input! Only Y (confirm) or ESC (cancel) is allowed.")

cv2 = None
mp = None
gesture_detection_enabled = False

ESC_KEY = 27
WINDOW_NAME = "Gesture Detection Camera"
VIDEO_RESOLUTION = (640, 360)
VIDEO_FRAME_RATE = 30.0
COLOR_GESTURE_BOX = (0, 255, 0)
COLOR_TEXT_DISPLAY = (0, 0, 255)
FONT_SCALE_SIZE = 0.6
FONT_THICKNESS = 2
FONT_TYPE = None

HAND_DETECTION_CONFIDENCE = 0.5
HAND_TRACKING_CONFIDENCE = 0.5
FINGER_THRESHOLD_ANGLE = 90

REQUIRED_DEPENDENCIES = {"opencv-python": "cv2", "mediapipe": "mediapipe"}
PIP_INSTALL_COMMAND = [sys.executable, "-m", "pip", "install"]
INSTALLATION_TIMEOUT = 300
INSTALLATION_LOCK = Lock()
INSTALLATION_PROGRESS = 0.0

def capture_keyboard_input():
    if not msvcrt.kbhit():
        return None
    key = msvcrt.getch()
    key_code = ord(key)
    if key_code == ESC_KEY:
        return "ESC"
    elif key_code == 13:
        return "ENTER"
    try:
        return key.decode("utf-8").strip()
    except (UnicodeDecodeError, AttributeError):
        return None

def stable_release_hardware_resources(capture=None):
    try:
        if capture and capture.isOpened():
            capture.release()
        if cv2 is not None:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
    except Exception as e:
        logger.error(f"Resource release error: {str(e)}")
    print("All hardware resources released.")

def get_available_camera_names():
    cameras = []
    failed_attempts = 0
    max_failed = 3
    max_checks = 10
    print("Detecting available cameras...")
    for camera_id in range(max_checks):
        capture = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if capture.isOpened():
            try:
                cam_name = capture.get(cv2.CAP_PROP_DEVICE_NAME) or f"Default Camera_{camera_id}"
            except Exception:
                cam_name = f"Default Camera_{camera_id}"
            cameras.append((camera_id, cam_name))
            capture.release()
            failed_attempts = 0
            continue
        capture = cv2.VideoCapture(camera_id)
        if capture.isOpened():
            cam_name = f"Compatibility Mode Camera_{camera_id}"
            cameras.append((camera_id, cam_name))
            capture.release()
            failed_attempts = 0
        else:
            failed_attempts += 1
            if failed_attempts >= max_failed:
                break
    if not cameras:
        print("No available cameras detected. Exiting program...")
        sys.exit(1)
    return cameras

def initialize_camera(camera_id):
    capture = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture = cv2.VideoCapture(camera_id)
        if not capture.isOpened():
            logger.error(f"Camera ID {camera_id} initialization failed")
            return None
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_RESOLUTION[0])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_RESOLUTION[1])
    capture.set(cv2.CAP_PROP_FPS, VIDEO_FRAME_RATE)
    return capture

def calculate_angle(p1, p2, p3):
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    if mag1 == 0 or mag2 == 0:
        return 0
    cos_angle = dot_product / (mag1 * mag2)
    cos_angle = max(min(cos_angle, 1), -1)
    angle = math.degrees(math.acos(cos_angle))
    return angle

def get_finger_status(hand_landmarks, image_shape):
    h, w, _ = image_shape
    landmarks = []
    for lm in hand_landmarks.landmark:
        x = int(lm.x * w)
        y = int(lm.y * h)
        landmarks.append((x, y))
    finger_status = [0, 0, 0, 0, 0]
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    index_mcp = landmarks[5]
    thumb_angle = calculate_angle(thumb_tip, thumb_mcp, index_mcp)
    if thumb_angle > FINGER_THRESHOLD_ANGLE and thumb_tip[0] > thumb_ip[0]:
        finger_status[0] = 1
    finger_tips = [8, 12, 16, 20]
    finger_dips = [6, 10, 14, 18]
    finger_mcps = [5, 9, 13, 17]
    for i in range(1, 5):
        angle = calculate_angle(landmarks[finger_tips[i-1]], landmarks[finger_dips[i-1]], landmarks[finger_mcps[i-1]])
        if angle > FINGER_THRESHOLD_ANGLE:
            finger_status[i] = 1
    return finger_status

def recognize_gesture(finger_status):
    up_count = sum(finger_status)
    if finger_status == [1, 0, 0, 0, 0]:
        return "Thumb Up"
    elif finger_status == [0, 1, 1, 0, 0]:
        return "V Sign"
    elif up_count == 1 and finger_status[1] == 1:
        return "Number 1"
    elif up_count == 2 and finger_status[1] == 1 and finger_status[2] == 1:
        return "Number 2"
    elif up_count == 3 and finger_status[1] == 1 and finger_status[2] == 1 and finger_status[3] == 1:
        return "Number 3"
    elif up_count == 4 and finger_status[1] == 1 and finger_status[2] == 1 and finger_status[3] == 1 and finger_status[4] == 1:
        return "Number 4"
    elif up_count == 5:
        return "Number 5"
    else:
        return "Unknown"

def detect_gesture(frame, hands_module):
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_module.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_min = w
            y_min = h
            x_max = 0
            y_max = 0
            for lm in hand_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y
            x_min = max(0, x_min - 20)
            y_min = max(0, y_min - 20)
            x_max = min(w, x_max + 20)
            y_max = min(h, y_max + 20)
            finger_status = get_finger_status(hand_landmarks, frame.shape)
            gesture_name = recognize_gesture(finger_status)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), COLOR_GESTURE_BOX, FONT_THICKNESS)
            text_y = max(y_min - 10, 20)
            cv2.putText(
                frame,
                gesture_name,
                (x_min, text_y),
                FONT_TYPE,
                FONT_SCALE_SIZE,
                COLOR_TEXT_DISPLAY,
                FONT_THICKNESS
            )
    return frame

def display_copyright_information():
    print("\n" + "="*60)
    print("Gesture Detection V1.0")
    print("Copyright (c) Akira Amatsume. All Rights Reserved")
    print("="*60)
    print("Operation: Press Enter to enable device | Press ESC to exit")
    while True:
        key = capture_keyboard_input()
        if key == "ESC":
            confirm_prompt = "Are you sure to exit? [Y/n] "
            input_buf = ""
            print(f"\n{confirm_prompt}{input_buf}", end="", flush=True)
            while True:
                if msvcrt.kbhit():
                    sub_key = msvcrt.getch()
                    sub_key_code = ord(sub_key)
                    if sub_key_code == 8 and input_buf:
                        input_buf = input_buf[:-1]
                        print(f"\r{confirm_prompt}{input_buf} ", end="", flush=True)
                    elif sub_key_code == 13:
                        if not input_buf.strip():
                            print("\nInvalid input! Enter Y (exit) or n (cancel)")
                            print(f"{confirm_prompt}", end="", flush=True)
                            input_buf = ""
                            continue
                        confirm = input_buf.strip()[0].upper()
                        if confirm == "Y":
                            print("\nExiting program...")
                            stable_release_hardware_resources()
                            sys.exit(0)
                        elif confirm == "N":
                            print("\nCancel exit. Returning to main menu...")
                            print("\n" + "="*60)
                            print("Gesture Detection V1.0")
                            print("Copyright (c) Akira Amatsume. All Rights Reserved")
                            print("="*60)
                            print("Operation: Press Enter to enable device | Press ESC to exit")
                            break
                        else:
                            print("\nInvalid input! Only Y/n is allowed.")
                            print(f"{confirm_prompt}", end="", flush=True)
                            input_buf = ""
                    else:
                        try:
                            char = sub_key.decode("utf-8").strip()
                            if char:
                                input_buf += char
                                print(f"\r{confirm_prompt}{input_buf}", end="", flush=True)
                        except UnicodeDecodeError:
                            pass
            continue
        elif key == "ENTER":
            return True
        elif key is not None:
            print("Invalid input! Press Enter to enable or ESC to exit.")

def select_camera_device():
    cameras = get_available_camera_names()
    print("\nAvailable Cameras:")
    print("-"*50)
    for idx, (cam_id, cam_name) in enumerate(cameras):
        print(f"[{idx}] ID: {cam_id} | Name: {cam_name}")
    print("-"*50)
    print("Operation: Enter number to select | Press ESC to return")
    input_buf = ""
    print("Enter camera number: ", end="", flush=True)
    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            key_code = ord(key)
            if key_code == ESC_KEY:
                print("\nReturning to main menu...")
                return None
            elif key_code == 13:
                if not input_buf.strip():
                    print("\nEmpty input! Enter a valid number or ESC to return.")
                    print("Enter camera number: ", end="", flush=True)
                    input_buf = ""
                    continue
                try:
                    selected_idx = int(input_buf.strip())
                    if 0 <= selected_idx < len(cameras):
                        return cameras[selected_idx][0]
                    else:
                        print(f"\nInvalid number! Only 0~{len(cameras)-1} is allowed.")
                except ValueError:
                    print("\nInvalid input! Only numbers are allowed.")
                input_buf = ""
                print("Enter camera number: ", end="", flush=True)
            elif key_code == 8 and input_buf:
                input_buf = input_buf[:-1]
                print(f"\rEnter camera number: {input_buf} ", end="", flush=True)
            elif key.isdigit():
                input_buf += key.decode("utf-8")
                print(f"\rEnter camera number: {input_buf}", end="", flush=True)
            else:
                print("\nInvalid input! Only numbers, Enter or ESC is allowed.")
                print("Enter camera number: ", end="", flush=True)
                input_buf = ""

def operate_camera_device(camera_id):
    hands_module = None
    global gesture_detection_enabled
    if gesture_detection_enabled:
        mp_hands = mp.solutions.hands
        hands_module = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=HAND_TRACKING_CONFIDENCE
        )
        if not hands_module:
            print("Gesture detection model load failed. Only basic camera will run.")
            gesture_detection_enabled = False
    capture = initialize_camera(camera_id)
    if not capture:
        return
    print(f"\nCamera {camera_id} started! Press ESC to stop and return to main menu.")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        while True:
            if capture_keyboard_input() == "ESC":
                print("\nStopping camera...")
                break
            ret, frame = capture.read()
            if not ret:
                logger.error("Failed to read camera frame")
                break
            if gesture_detection_enabled and hands_module:
                frame = detect_gesture(frame, hands_module)
            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(10)
    except Exception as e:
        logger.error(f"Camera operation error: {str(e)}")
    finally:
        if hands_module:
            hands_module.close()
        stable_release_hardware_resources(capture)

def main():
    global cv2, mp, gesture_detection_enabled, FONT_TYPE
    cv2, mp, gesture_detection_enabled = safe_check_and_install_dependencies()
    if cv2 is None or mp is None:
        print("Critical error: cv2/mediapipe module not available. Exiting program...")
        sys.exit(1)
    FONT_TYPE = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        if not display_copyright_information():
            continue
        camera_id = select_camera_device()
        if camera_id is None:
            continue
        operate_camera_device(camera_id)
        print("Returning to main menu...\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Program fatal error: {str(e)}")
    finally:
        stable_release_hardware_resources()
