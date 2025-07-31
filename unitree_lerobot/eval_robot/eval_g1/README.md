# G1 机器人评估模块

本目录包含了用于在真实 G1 机器人上评估训练好的策略的完整系统。

## 目录结构

```
eval_g1/
├── README.md                    # 本文档
├── eval_g1.py                   # 真实机器人评估主程序
├── eval_g1_dataset.py           # 数据集评估程序（仿真环境）
├── eval_real_config.py          # 评估配置类
├── image_server/                # 图像服务器模块
│   ├── image_server.py          # 图像服务器
│   └── image_client.py          # 图像客户端
└── robot_control/               # 机器人控制模块
    ├── robot_arm.py             # G1 机械臂控制器
    └── robot_hand_unitree.py    # 灵巧手控制器
```

## 功能概述

### 1. 真实机器人评估 (`eval_g1.py`)

这是主要的评估程序，用于在真实的 G1 机器人上运行训练好的策略。

**主要特性：**
- 实时图像采集和处理
- 机械臂和灵巧手的同步控制
- 支持双目视觉和腕部相机
- 基于共享内存的高效图像传输
- 50Hz 控制频率

**支持的硬件配置：**
- **机械臂**: G1-29 双臂系统
- **灵巧手**: Dex3.1 灵巧手 或 夹爪
- **相机**: 头部双目相机 + 腕部相机（可选）

### 2. 数据集评估 (`eval_g1_dataset.py`)

用于在仿真环境中评估策略性能，可以：
- 从数据集中读取观测数据
- 预测动作并与真实动作比较
- 生成预测动作与真实动作的对比图表（保存为 `figure.png`）
- 可选择是否将动作发送到真实机器人（通过设置 `send_real_robot` 参数）

**评估流程：**
1. 加载指定的数据集和策略模型
2. 从数据集中读取观测数据（图像和状态）
3. 使用策略模型预测动作
4. 将预测动作与数据集中的真实动作进行比较
5. 生成可视化图表，显示每个动作维度的对比结果
6. 可选择将预测动作发送到真实机器人执行（用于验证泛化能力）

**输出文件：**
- `figure.png`: 包含所有动作维度的预测vs真实动作对比图表

### 3. 图像服务器 (`image_server/`)

提供实时图像采集和处理功能：
- 支持多种相机类型（OpenCV、RealSense等）
- 自动处理双目图像拼接
- 基于共享内存的高效数据传输

### 4. 机器人控制 (`robot_control/`)

提供底层机器人控制接口：
- G1 机械臂的关节控制
- Dex3.1 灵巧手的精细控制
- 夹爪的开合控制

## 使用方法

### 环境准备

1. **硬件要求：**
   - G1-29 双臂机器人
   - Dex3.1 灵巧手 或 夹爪
   - 头部双目相机
   - 腕部相机（可选）

2. **软件依赖：**
   ```bash
   pip install torch lerobot numpy opencv-python
   ```

3. **相机配置：**
   - 确保相机驱动正确安装
   - 配置相机ID和分辨率

### 运行真实机器人评估

1. **启动图像服务器：**
   ```bash
   python image_server/image_server.py
   ```

2. **运行评估程序：**
   ```bash
   python eval_g1.py --repo_id <数据集ID> --policy.path <策略路径>
   ```

   **参数说明：**
   - `--repo_id`: 数据集仓库ID，格式为 `{hf_username}/{dataset_name}`（例如：`lerobot/test`）
   - `--policy.path`: 预训练策略路径，可以是：
     - 本地目录路径：包含 `config.json` 和 `model.safetensors` 的目录
     - Hugging Face Hub 仓库ID：格式为 `{hf_username}/{model_name}`
     - 如果不提供此参数，将使用随机初始化的策略（仅用于调试）

   **示例：**
   ```bash
   # 使用本地策略文件
   python eval_g1.py --repo_id lerobot/test --policy.path ./outputs/train/my_smolvla/checkpoint_000200

   # 使用Hub上的策略
   python eval_g1.py --repo_id lerobot/test --policy.path lerobot/act_aloha_sim_transfer_cube_human
   ```

3. **程序流程：**
   - 程序会提示输入 's' 开始执行
   - 机器人会移动到初始位置
   - 开始实时策略执行循环

### 配置参数

#### 图像配置
```python
img_config = {
    'fps': 30,                                    # 图像帧率
    'head_camera_type': 'opencv',                 # 头部相机类型
    'head_camera_image_shape': [480, 1280],       # 头部相机分辨率
    'head_camera_id_numbers': [0],                # 头部相机ID
    'wrist_camera_type': 'opencv',                # 腕部相机类型
    'wrist_camera_image_shape': [480, 640],       # 腕部相机分辨率
    'wrist_camera_id_numbers': [2, 4],            # 腕部相机ID
}
```

#### 机器人配置
```python
robot_config = {
    'arm_type': 'g1',                             # 机械臂类型
    'hand_type': "dex3",                          # 手部类型：'dex3' 或 'gripper'
}
```

### 运行数据集评估

```bash
python eval_g1_dataset.py --repo_id <数据集ID> --policy.path <策略路径>
```

**参数说明：**
- `--repo_id`: 数据集仓库ID，格式为 `{hf_username}/{dataset_name}`（例如：`lerobot/test`）
- `--policy.path`: 预训练策略路径，可以是：
  - 本地目录路径：包含 `config.json` 和 `model.safetensors` 的目录
  - Hugging Face Hub 仓库ID：格式为 `{hf_username}/{model_name}`
  - 如果不提供此参数，将使用随机初始化的策略（仅用于调试）

**示例：**
```bash
# 使用本地策略文件
python eval_g1_dataset.py --repo_id lerobot/test --policy.path ./outputs/train/my_smolvla/checkpoint_000200

# 使用Hub上的策略
python eval_g1_dataset.py --repo_id lerobot/test --policy.path lerobot/act_aloha_sim_transfer_cube_human

# 使用随机初始化策略（调试模式）
python eval_g1_dataset.py --repo_id lerobot/test
```

## 注意事项

### 参数验证
- **数据集ID格式**: 必须符合 `{hf_username}/{dataset_name}` 格式
- **策略路径**: 如果提供本地路径，确保目录包含完整的模型文件（`config.json` 和 `model.safetensors`）
- **网络连接**: 使用Hub上的数据集或策略时，需要确保网络连接正常
- **权限检查**: 确保对指定的数据集和策略有访问权限

### 安全警告
⚠️ **重要安全提醒：**
- 运行前确保机器人周围有足够的安全空间
- 确保紧急停止按钮可随时使用
- 建议在首次运行时使用较低的控制频率
- 密切监控机器人运动，准备随时停止

### 技术细节

1. **图像处理：**
   - 自动检测双目相机配置
   - 支持不同分辨率的图像拼接
   - 实时图像格式转换

2. **控制频率：**
   - 默认控制频率：50Hz
   - 可通过修改 `frequency` 变量调整

3. **共享内存：**
   - 使用共享内存进行图像数据传输
   - 支持多进程间的数据共享
   - 自动管理内存分配和释放

4. **动作执行：**
   - 机械臂：14维关节角度控制
   - 灵巧手：7维关节角度控制（每只手）
   - 夹爪：1维开合控制（每只手）

## 故障排除

### 常见问题

1. **数据集加载失败：**
   - 检查数据集ID格式是否正确（应为 `{hf_username}/{dataset_name}`）
   - 确认网络连接正常（如果使用Hub上的数据集）
   - 验证对数据集有访问权限
   - 检查数据集是否包含所需的观测数据格式

2. **策略模型加载失败：**
   - 检查策略路径是否正确
   - 确认本地目录包含完整的模型文件（`config.json` 和 `model.safetensors`）
   - 验证Hub上的策略模型是否存在且可访问
   - 检查模型文件是否损坏

3. **相机连接失败：**
   - 检查相机ID配置
   - 确认相机驱动安装正确
   - 验证相机权限

4. **机器人连接失败：**
   - 检查网络连接
   - 确认机器人IP地址配置
   - 验证控制权限

5. **图像显示异常：**
   - 检查图像分辨率配置
   - 确认双目相机拼接参数
   - 验证共享内存分配

6. **控制延迟：**
   - 降低控制频率
   - 检查网络延迟
   - 优化图像处理流程

### 调试模式

启用详细日志输出：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 扩展开发

### 添加新的相机类型

1. 在 `image_server.py` 中添加新的相机类
2. 实现标准的图像采集接口
3. 更新配置参数

### 添加新的手部类型

1. 在 `robot_hand_unitree.py` 中添加新的控制器类
2. 实现标准的控制接口
3. 更新机器人配置

### 自定义评估指标

1. 修改 `eval_policy` 函数
2. 添加性能指标计算
3. 实现结果保存和可视化

## 相关文档

- [LeRobot 官方文档](https://lerobot.github.io/)
- [G1 机器人技术手册](https://www.unitree.com/products/g1)
- [Dex3.1 灵巧手文档](https://www.unitree.com/products/dex3)

## 许可证

本项目遵循 LeRobot 项目的许可证条款。 