import time
import mujoco
import mujoco.viewer
import zmq

model = mujoco.MjModel.from_xml_path('../models/mjcf/R7-900_description.xml')
data = mujoco.MjData(model)

# conifg zmq server
ctx = zmq.Context()
sock = ctx.socket(zmq.REP)
sock.bind("tcp://*:5555")

print("MuJoCo Simulation Server Started. Waiting for ZMQ commands...")

with mujoco.viewer.launch_passive(model,data) as viewer:
    while viewer.is_running():
        # 1. 接收 C++ 发来的指令 (阻塞等待)
        try:
            message = sock.recv_json(flags=zmq.NOBLOCK) # 使用非阻塞接收，以免卡死 viewer 刷新
        except zmq.Again:
            #如果没有新指令，则维持上一帧的 ctrl 继续 step
            mujoco.mj_step(model, data)
            viewer.sync()
            continue

        # 2. 解析指令并下发控制
        # 兼容两种格式：
        # 1. 纯列表: [q1, q2, ...]
        # 2. 字典(对应 C++ HardJointCmd): {"positions": [...], "velocities": [...], ...}
        target_pos = None
        
        if isinstance(message, list) and len(message) == 6:
            target_pos = message
        elif isinstance(message, dict) and "positions" in message:
            target_pos = message["positions"]
        
        if target_pos is not None and len(target_pos) == 6:
            data.ctrl[:] = target_pos
        else:
            print(f"Warning: Invalid message format: {message}")
        
        # 3. 执行物理仿真
        # 收到一个指令，通常意味着执行一个控制周期
        # 如果 C++ 端是 1kHz 发送，这里就 step 一次 (0.001s)
        mujoco.mj_step(model, data)
        
        # 4. 回复确认信息 (ACK)
        # C++ 端不处理状态反馈，但 ZMQ REQ-REP 模式必须有回复才能进行下一次发送
        sock.send_string("ACK")

        # 5. 刷新画面
        viewer.sync()
