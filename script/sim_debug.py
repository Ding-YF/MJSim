import mujoco.viewer
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, '../models/mjcf/franka_emika_panda/scene.xml')
print(f"Loading model from: {xml_path}")

try:
    mujoco.viewer.launch_from_path(xml_path)
except Exception as e:
    print(f"Failed to launch viewer: {e}")