import mujoco
import mujoco.viewer

# 定义模型路径配置
MODEL_PATHS = {
    'aglity': '../models/mjcf/agility_cassie/cassie.xml',
    'boston_dynamics': '../models/mjcf/boston_dynamics_spot/spot.xml',
    'unitree': '../models/mjcf/unitree_g1/g1_mjx.xml',
    'panda': '../models/mjcf/franka_emika_panda/mjx_panda.xml',
    'ur5': '../models/mjcf/universal_robots_ur5e/ur5e.xml',
    'QJ': '../models/mjcf/R7-900_description.xml',
}

# 选择要使用的模型
current_model = 'QJ'  # 可选 'panda', 'ur5', 'shadow_hand', 'QJ'

model = mujoco.MjModel.from_xml_path(MODEL_PATHS[current_model])
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model,data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync() 