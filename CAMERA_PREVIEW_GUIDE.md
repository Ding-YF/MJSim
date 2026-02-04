# 相机预览功能使用指南

## 功能说明

仿真器现在支持实时显示两个相机的画面：
- **Scene Camera (环境相机)**: 右前方45度俯视视角
- **Hand Camera (腕部相机)**: 安装在机械臂手腕上的第一人称视角

## 安装依赖

相机预览功能需要 OpenCV：

```bash
pip install opencv-python
```

## 使用方法

### 1. 基本用法（无预览，部署模式）

```bash
python script/openpi_zmqclient.py
```

### 2. 启用相机预览（调试模式）

```bash
python script/openpi_zmqclient.py --preview
```

### 3. 自定义预览帧率

```bash
# 默认15FPS，可以调整以平衡性能和流畅度
python script/openpi_zmqclient.py --preview --preview-fps 30
```

### 4. 指定服务器地址

```bash
python script/openpi_zmqclient.py --server tcp://192.168.1.100:5555 --preview
```

### 5. 完整参数示例

```bash
python script/openpi_zmqclient.py \
    --server tcp://localhost:5555 \
    --preview \
    --preview-fps 20 \
    --model ../models/mjcf/franka_emika_panda/scene.xml
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--server` | 策略服务器地址 | `tcp://localhost:5555` |
| `--preview` | 启用相机预览窗口 | 关闭 |
| `--preview-fps` | 预览窗口刷新率 | 15 FPS |
| `--model` | MuJoCo 模型文件路径 | `../models/mjcf/franka_emika_panda/scene.xml` |

## 操作说明

- **预览窗口控制**: 在预览窗口按 `q` 键可关闭预览（仿真继续运行）
- **退出仿真**: 按 `Ctrl+C` 或关闭 MuJoCo viewer 窗口