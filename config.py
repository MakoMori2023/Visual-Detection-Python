import yaml
import os
from typing import Dict, Any

CONFIG: Dict[str, Any] = {}

def load_config(config_path: str = None) -> Dict[str, Any]:
    global CONFIG
    if not config_path:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config isn't exit: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f)
    
    _complete_model_paths()
    
    return CONFIG

def _complete_model_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if "face_detection" in CONFIG and "model_path" in CONFIG["face_detection"]:
        rel_path = CONFIG["face_detection"]["model_path"]
        CONFIG["face_detection"]["model_path"] = os.path.join(current_dir, rel_path)

    if "hand_gesture_detection" in CONFIG and "model_path" in CONFIG["hand_gesture_detection"]:
        rel_path = CONFIG["hand_gesture_detection"]["model_path"]
        CONFIG["hand_gesture_detection"]["model_path"] = os.path.join(current_dir, rel_path)

    if "human_stickman_detection" in CONFIG and "model_path" in CONFIG["human_stickman_detection"]:
        rel_path = CONFIG["human_stickman_detection"]["model_path"]
        CONFIG["human_stickman_detection"]["model_path"] = os.path.join(current_dir, rel_path)

load_config()