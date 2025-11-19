import mujoco
import os
import re

# 配置路径
script_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.abspath(os.path.join(script_dir, "../models/urdf/QJ7-900_stl/urdf/R7-900_description.urdf"))
output_path = os.path.abspath(os.path.join(script_dir, "../models/mjcf/R7-900_description.xml"))
temp_urdf_path = os.path.abspath(os.path.join(script_dir, "../models/urdf/QJ7-900_stl/urdf/R7-900_description_mujoco.urdf"))

print(f"读取原始 URDF 文件: {urdf_path}")

# 读取并修改 URDF 文件
with open(urdf_path, 'r', encoding='utf-8') as f:
    urdf_content = f.read()

# 需要加上<mujoco>标签，告诉Mujoco网格文件的位置
mujoco_config = '''  <mujoco>
    <compiler meshdir="../meshes/" balanceinertia="true" discardvisual="false"/>
  </mujoco>
'''

# 在 <robot> 标签后添加 mujoco 配置
urdf_content = re.sub(
    r'(<robot[^>]*>)',
    r'\1\n' + mujoco_config,
    urdf_content
)

# 保存修改后的 URDF
with open(temp_urdf_path, 'w', encoding='utf-8') as f:
    f.write(urdf_content)

print(f"保存修改后的 URDF: {temp_urdf_path}")

try:
    # 从修改后的 URDF 文件加载模型
    model = mujoco.MjModel.from_xml_path(temp_urdf_path)
    
    # 保存为 MJCF XML 格式
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mujoco.mj_saveLastXML(output_path, model)
    
    print(f"转换成功！MJCF 文件已保存到: {output_path}")
    
except Exception as e:
    print(f"转换失败: {e}")
finally:
    # 清理临时文件
    if os.path.exists(temp_urdf_path):
        os.remove(temp_urdf_path)
