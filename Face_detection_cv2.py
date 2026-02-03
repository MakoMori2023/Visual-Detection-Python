import sys
import subprocess
import logging
import time
import msvcrt
from threading import Thread, Lock

logging.basicConfig(
    level=logging.ERROR,
    format="Abnormal termination File:%(filename)s | Line:%(lineno)d | Error message:%(message)s",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

def check_missing_dependencies():
    required_deps = {"opencv-python": "cv2"}
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
    face_detection_enabled = False

    if not missing_deps:
        import cv2
        cv2_module = cv2
        face_detection_enabled = True
        print("All required dependencies are installed. Face detection enabled.")
        return cv2_module, face_detection_enabled

    print(f"Missing dependencies: {', '.join(missing_deps)}")
    print("Face detection requires these dependencies. Install now?")
    print("Press Y to confirm, ESC to cancel (case-sensitive)")

    while True:
        key = capture_keyboard_input()
        if key == "ESC":
            print("\nInstallation cancelled. Only basic camera function (if available) will run.")
            try:
                import cv2
                cv2_module = cv2
            except ImportError:
                cv2_module = None
            return cv2_module, False

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
                        cv2_module = cv2
                        face_detection_enabled = True
                        print("All dependencies installed. Face detection enabled.")
                    else:
                        print(f"Installation completed but missing: {', '.join(remaining_missing)}")
                else:
                    print(f"Installation failed (return code: {process.returncode})")

            except subprocess.TimeoutExpired:
                process.kill()
                progress_thread.join()
                print(f"Installation timed out (>{INSTALLATION_TIMEOUT}s)")

            return cv2_module, face_detection_enabled

        elif key is not None:
            print("Invalid input! Only Y (confirm) or ESC (cancel) is allowed.")

cv2 = None
face_detection_enabled = False

ESC_KEY = 27
WINDOW_NAME = "Face Detection Camera"
VIDEO_RESOLUTION = (640, 360)
VIDEO_FRAME_RATE = 30.0
COLOR_SCANNING = (255, 0, 0)
COLOR_TEXT_DISPLAY = (0, 0, 255)
FONT_SCALE_SIZE = 0.6
FONT_THICKNESS = 2
FONT_TYPE = None

REQUIRED_DEPENDENCIES = {"opencv-python": "cv2"}
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

def detect_human_face(frame, face_cascade):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for idx, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x+w, y+h), COLOR_SCANNING, FONT_THICKNESS)
        text_y = max(y - 10, 20)
        cv2.putText(
            frame,
            f"Person {idx+1}",
            (x, text_y),
            FONT_TYPE,
            FONT_SCALE_SIZE,
            COLOR_TEXT_DISPLAY,
            FONT_THICKNESS
        )

    return frame

def display_copyright_information():
    print("\n" + "="*60)
    print("Face Detection V1.0")
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
                            print("Face Detection V1.0")
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
    face_cascade = None
    global face_detection_enabled
    if face_detection_enabled:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            print("Face detection model load failed. Only basic camera will run.")
            face_detection_enabled = False

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

            if face_detection_enabled and face_cascade:
                frame = detect_human_face(frame, face_cascade)

            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(10)

    except Exception as e:
        logger.error(f"Camera operation error: {str(e)}")
    finally:
        stable_release_hardware_resources(capture)

def main():
    global cv2, face_detection_enabled, FONT_TYPE

    cv2, face_detection_enabled = safe_check_and_install_dependencies()
    if cv2 is None:
        print("Critical error: cv2 module not available. Exiting program...")
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
