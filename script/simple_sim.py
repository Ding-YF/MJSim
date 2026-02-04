import mujoco
import mujoco.viewer
import os

# 定义模型路径配置
MODEL_PATHS = {
    'aglity': '../models/mjcf/agility_cassie/cassie.xml',
    'boston_dynamics': '../models/mjcf/boston_dynamics_spot/spot.xml',
    'unitree': '../models/mjcf/unitree_g1/g1_mjx.xml',
    'panda': '../models/mjcf/franka_emika_panda/mjx_panda.xml',
    'panda_scene': '../models/mjcf/franka_emika_panda/scene.xml',
    'ur5': '../models/mjcf/universal_robots_ur5e/ur5e.xml',
    'QJ': '../models/mjcf/R7-900_description.xml',
}

# 选择要使用的模型
current_model = 'panda_scene'  # 可选 'panda', 'ur5', 'shadow_hand', 'QJ'

# 获取脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 构建模型的绝对路径
model_path = os.path.join(script_dir, MODEL_PATHS[current_model])

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model,data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync() 