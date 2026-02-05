import time
import os
import threading
import numpy as np
import mujoco
import mujoco.viewer
import zmq
import msgpack
import msgpack_numpy
from collections import deque
import cv2
import tyro
from dataclasses import dataclass

# Patch msgpack to use numpy serialization
msgpack_numpy.patch()

# 全局定义的初始位姿 (Joint1 - Joint7)
# User Request: Joint4 = -45 deg, Joint6 = 60 deg
DEFAULT_INIT_POSE = np.array([0, 0, 0, np.deg2rad(-45), 0, np.deg2rad(60), 0])

@dataclass
class SimulationConfig:
    """MuJoCo Simulator with OpenPI Policy Client"""
    
    server: str = "tcp://localhost:5555"
    """Policy server address"""
    
    preview: bool = False
    """Enable camera preview windows (requires OpenCV)"""
    
    preview_fps: int = 15
    """Camera preview frame rate"""
    
    model: str = "../models/mjcf/franka_emika_panda/scene.xml"
    """Path to MuJoCo model XML file"""
    
    sim_dt: float = 0.002
    """Simulation timestep in seconds (default: 0.002s / 500Hz)"""

    prompt: str = "Pick up the cube and then place it in the box."
    """Task description prompt"""

class ZMQPolicyClient:
    """
    封装 ZMQ 通信细节，连接管理和静默重连
    """
    def __init__(
        self, 
        server_addr: str = "tcp://localhost:5555",
        recv_timeout_ms: int = 2000,
        reconnect_interval_s: float = 3.0,
        silent_mode: bool = True
    ):
        self.server_addr = server_addr
        self.recv_timeout_ms = recv_timeout_ms
        self.reconnect_interval_s = reconnect_interval_s
        self.silent_mode = silent_mode  # 是否静默模式（减少日志输出）
        
        self.ctx = None
        self.socket = None
        self.metadata = {}
        
        # 连接状态管理
        self.is_connected = False
        self.last_reconnect_attempt = 0.0  # 上次重连尝试的时间
        self.connection_warned = False  # 是否已经警告过连接失败

    def connect(self):
        """在工作线程中调用此方法建立连接"""
        if not self.silent_mode:
            print(f"[Client] Connecting to Policy Server at {self.server_addr}...")
            
        self.ctx = zmq.Context()
        self._create_socket()
        
        # 尝试握手验证连接
        if self._try_handshake():
            self.is_connected = True
            if not self.silent_mode or not self.connection_warned:
                print(f"[Client] ✓ Connected to server successfully")
                if "policy_name" in self.metadata:
                    print(f"[Client] Policy: {self.metadata['policy_name']}")
            self.connection_warned = False
        else:
            self.is_connected = False
            if not self.connection_warned:
                print(f"[Client] ✗ Server not responding, will retry silently in background...")
                self.connection_warned = True

    def _create_socket(self):
        """创建并配置 socket"""
        if self.socket:
            self.socket.close()
            
        self.socket = self.ctx.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.recv_timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self.server_addr)

    def _try_handshake(self) -> bool:
        """尝试握手，返回是否成功"""
        try:
            self.socket.send(b"metadata")
            self.metadata = msgpack.unpackb(self.socket.recv())
            return True
        except (zmq.Again, zmq.ZMQError):
            # 握手失败，需要重置 socket（REQ-REP 模式要求）
            self._create_socket()
            return False

    def _maybe_reconnect(self):
        """根据重连间隔尝试重新连接"""
        current_time = time.time()
        if current_time - self.last_reconnect_attempt >= self.reconnect_interval_s:
            self.last_reconnect_attempt = current_time
            if self._try_handshake():
                self.is_connected = True
                print(f"[Client] ✓ Reconnected to server successfully")
                self.connection_warned = False

    def close(self):
        if self.socket:
            self.socket.close()
        if self.ctx:
            self.ctx.term()

    def infer(self, obs: dict) -> np.ndarray:
        """发送观测，返回动作块 (Chunk)"""
        if self.socket is None:
            return np.array([])
        
        # 如果未连接，尝试静默重连
        if not self.is_connected:
            self._maybe_reconnect()
            if not self.is_connected:
                return np.array([])
            
        try:
            self.socket.send(msgpack.packb(obs))
            result = msgpack.unpackb(self.socket.recv())
            
            if "actions" in result:
                # 确保返回的是 numpy 数组
                actions = np.array(result["actions"])
                # 如果是 (Batch, Time, Dim) 这种 3D 数组 (Batch=1)，降维
                if actions.ndim == 3 and actions.shape[0] == 1:
                    actions = actions[0]
                return actions
            return np.array([])
            
        except zmq.Again:
            # 超时，标记为未连接，静默处理
            self.is_connected = False
            self._create_socket()  # 重置 socket 状态
            if not self.silent_mode:
                print("[Client] Server timeout, waiting for reconnection...")
            return np.array([])
            
        except zmq.ZMQError as e:
            # ZMQ 错误，标记为未连接
            self.is_connected = False
            self._create_socket()
            if not self.silent_mode:
                print(f"[Client] ZMQ Error: {e}, waiting for reconnection...")
            return np.array([])

class AsyncActionBroker:
    """
    异步动作代理。
    在后台线程中进行网络请求，保证主仿真循环不卡顿。
    """
    def __init__(
        self, 
        server_addr: str,
        silent_mode: bool = True,
        reconnect_interval_s: float = 3.0,
        action_watermark: int = 25
    ):
        self.client = ZMQPolicyClient(
            server_addr=server_addr,
            silent_mode=silent_mode,
            reconnect_interval_s=reconnect_interval_s
        )
        self.action_watermark = action_watermark
        self.queue = deque()
        self.lock = threading.Lock()
        
        # 线程控制
        self.running = True
        self.request_event = threading.Event() # 通知后台线程：有新观测，准备去请求
        self.current_obs = None
        
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        
    def _worker_loop(self):
        """后台线程：负责建立连接和处理网络请求"""
        # 在线程内部建立连接，避免跨线程上下文问题
        self.client.connect()
        
        while self.running:
            # 1. 如果未连接，尝试定期重连
            if not self.client.is_connected:
                time.sleep(0.5)  # 未连接时降低循环频率
                self.client._maybe_reconnect()
                continue
            
            # 2. 已连接，等待前端请求信号
            if self.request_event.wait(timeout=0.1):
                # 收到信号，执行推理
                if self.current_obs is not None:
                    start_t = time.time()
                    actions = self.client.infer(self.current_obs)
                    dt = time.time() - start_t
                    
                    if len(actions) > 0:
                        print(f"[Broker] Fetched {len(actions)} actions in {dt*1000:.1f}ms")
                        with self.lock:
                            self.queue.extend(actions)
                    
                    self.current_obs = None # 清空已消费的观测
                    self.request_event.clear() # 重置信号

    def is_connected(self) -> bool:
        """返回当前是否已连接到策略服务器"""
        return self.client.is_connected

    def stop(self):
        self.running = False
        self.thread.join()
        self.client.close()

    def get_action(self, obs_callback) -> np.ndarray | None:
        """
        非阻塞获取动作
        只有在已连接且队列为空时，才会调用 obs_callback 获取观测并触发后台推理
        """
        # 0. 如果未连接到服务器，直接返回 None（不触发观测采集）
        if not self.client.is_connected:
            return None
        
        # 1. 检查是否需要触发新请求 (队列空 且 当前没有正在进行的请求)
        # 这里使用 len(queue) == 0 作为触发条件
        # 也可以设置一个水位线 (例如剩余 5 个动作时就提前请求)
        with self.lock:
            q_len = len(self.queue)
        
        if q_len <= self.action_watermark and not self.request_event.is_set():
            # 获取观测
            # print(f"[Broker] Queue low ({q_len} <= {self.action_watermark}), capturing observation...")
            obs = obs_callback()
            self.current_obs = obs
            self.request_event.set() # 唤醒后台线程

        # 2. 尝试取出动作
        with self.lock:
            if self.queue:
                return self.queue.popleft()
            
        return None

def run_simulation(config: SimulationConfig):
    # 1. 加载 Mujoco 模型
    model_path = config.model
    if not os.path.exists(model_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_path)
        
    print(f"[Sim] Loading model from: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    
    # 设置仿真步长，即控制频率
    if config.sim_dt is not None:
        model.opt.timestep = config.sim_dt
        print(f"[Sim] Simulation timestep set to: {config.sim_dt}s ({1.0/config.sim_dt:.1f}Hz)")
        
    data = mujoco.MjData(model)

    # 2. 初始化多相机渲染器
    # 注意：两个相机都是 640x480 分辨率
    renderer_scene = mujoco.Renderer(model, height=480, width=640)
    renderer_hand = mujoco.Renderer(model, height=480, width=640)

    # 3. 初始化 异步 Broker (Client 在后台线程启动，不会卡住这里)
    broker = AsyncActionBroker(config.server)

    # 4. 定义获取观测的函数
    def capture_observation():
        # A. 图像采集（两个相机）
        # 场景相机（环境视角）
        renderer_scene.update_scene(data, camera="scene_camera")
        img_scene = renderer_scene.render()
        
        # 手部相机（腕部视角）
        renderer_hand.update_scene(data, camera="hand_camera")
        img_hand = renderer_hand.render()
        
        # B. 状态
        # qpos[:7] contains 7 arm joints. qpos[7], qpos[8] are gripper fingers (0-0.04m).
        qpos_arm = data.qpos[:7].astype(np.float32)
        gripper_width = data.qpos[7] + data.qpos[8]
        gripper_state = gripper_width / 0.08 # Normalize 0-1 (1=Open)
        
        # 适配 Pi05 DROID 的输入格式
        # DROID policy 期望特定的键名：observation/joint_position, observation/gripper_position 等
        
        return {
            "observation/joint_position": qpos_arm,  # 7 DoF Arm
            "observation/gripper_position": np.array([gripper_state], dtype=np.float32), # 1 dim
            "observation/exterior_image_1_left": img_scene, # 环境相机
            "observation/wrist_image_left": img_hand,       # 手腕相机
            "prompt": config.prompt
        }

    # 5. 相机预览设置, 可在run_simulation参数中控制
    preview_enabled = config.preview
    
    if preview_enabled:
        print(f"[Sim] Camera preview enabled at {config.preview_fps} FPS")
        cv2.namedWindow("Scene Camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Hand Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Scene Camera", 640, 480)
        cv2.resizeWindow("Hand Camera", 640, 480)
        preview_interval = 1.0 / config.preview_fps  # 预览更新间隔
        last_preview_time = 0.0
    
    # 6. 仿真循环
    dt = model.opt.timestep
    
    last_connected_state = None

    # 初始化机器人姿态
    print("[Sim] Initializing robot pose...")
    # 1. 设置物理状态 (直接修改 qpos 瞬间瞬移)
    # 根据模型定义，Panda 的前 7 个关节对应 qpos 的前 7 位
    data.qpos[:7] = DEFAULT_INIT_POSE
    
    # 2. 设置控制指令 (修改 ctrl)
    # 必须同时更新 actuator 的输入，否则位置控制器的 PID 误差会试图把机器人瞬间拉回零位
    data.ctrl[:7] = DEFAULT_INIT_POSE
    
    # 刷新动力学状态，确保位置生效
    mujoco.mj_forward(model, data)

    # 初始化当前动作为初始位姿 (而不是全0)，实现 Zero-Order Hold
    current_action = np.zeros(model.nu) 
    current_action[:7] = DEFAULT_INIT_POSE
    
    input("[Note] Press any key to start simulation...\n")
    
    print("[Sim] Starting simulation loop... Press Ctrl+C to stop.")
    print("[Sim] Physics simulation is running, control will activate when server connects.")
    if preview_enabled:
        print("[Sim] Press 'q' on camera windows to close preview (simulation continues).")
    
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                
                # --- 连接状态监控 (只在状态变化时打印) ---
                current_connected = broker.is_connected()
                if current_connected != last_connected_state:
                    if current_connected:
                        print("[Sim] ✓ Policy control activated")
                    else:
                        if last_connected_state is not None:  # 不是第一次检查
                            print("[Sim] ⏸ Policy control paused (server disconnected)")
                    last_connected_state = current_connected
                
                # --- 核心逻辑: 异步获取动作 ---
                next_action = broker.get_action(capture_observation)
                
                if next_action is not None:
                    # 有新动作，更新控制
                    # 假定 Action 是 [J1...J7, Gripper]
                    
                    # 取前7个维度作为关节控制 (Panda是7轴)
                    n_joints = 7
                    if len(next_action) >= n_joints:
                         current_action[:n_joints] = next_action[:n_joints]
                    
                    # 处理夹爪 (最后一个维度)
                    g_cmd = next_action[-1]
                    g_val = 255.0 if g_cmd > 0.5 else 0.0
                    
                    # Apply to all remaining actuators (assuming they are gripper)
                    for i in range(n_joints, model.nu):
                        current_action[i] = g_val
                
                # --- 物理执行 ---
                data.ctrl[:] = current_action
                mujoco.mj_step(model, data)
                viewer.sync()
                
                # --- 相机预览更新 (按设定的FPS更新) ---
                if preview_enabled:
                    current_time = time.time()
                    if current_time - last_preview_time >= preview_interval:
                        # 渲染场景相机
                        renderer_scene.update_scene(data, camera="scene_camera")
                        img_scene = renderer_scene.render()
                        # RGB -> BGR (OpenCV格式)
                        img_scene_bgr = cv2.cvtColor(img_scene, cv2.COLOR_RGB2BGR)
                        cv2.imshow("Scene Camera", img_scene_bgr)
                        
                        # 渲染手部相机
                        renderer_hand.update_scene(data, camera="hand_camera")
                        img_hand = renderer_hand.render()
                        img_hand_bgr = cv2.cvtColor(img_hand, cv2.COLOR_RGB2BGR)
                        cv2.imshow("Hand Camera", img_hand_bgr)
                        
                        # 检查按键（1ms等待，避免阻塞）
                        key = cv2.waitKey(1)
                        if key == ord('q'):
                            print("[Sim] Closing camera preview windows...")
                            cv2.destroyAllWindows()
                            preview_enabled = False
                        
                        last_preview_time = current_time
                
                # 实时同步
                time_until_next = dt - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)
    finally:
        broker.stop()
        if preview_enabled:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    run_simulation(tyro.cli(SimulationConfig))
