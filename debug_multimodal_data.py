#!/usr/bin/env python3

import torch
import numpy as np
import sys
import os

# 切换到正确的工作目录
os.chdir('/home/lin/youkechuiguo/youkechuiguo_robot/unitree_IL_lerobot')
sys.path.insert(0, '/home/lin/youkechuiguo/youkechuiguo_robot/unitree_IL_lerobot')

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
import matplotlib.pyplot as plt

def analyze_multimodal_data():
    """分析多模态数据处理流程，重点检查视频和状态数据"""
    
    print("🔍 分析多模态数据处理流程")
    print("=" * 80)
    
    # 加载数据集
    print("📂 加载数据集...")
    dataset = LeRobotDataset("/home/lin/youkechuiguo/dataset/G1_ObjectPlacement_Dataset")
    
    print(f"数据集总长度: {len(dataset)}")
    print(f"数据集特征: {list(dataset.features.keys())}")
    
    # 分析不同样本的数据
    print("\n🔍 分析不同样本的多模态数据")
    print("-" * 60)
    
    # 选择来自不同episode的样本
    test_indices = [0, 320, 642, 960]  # 不同episode的起始点
    
    for i, idx in enumerate(test_indices):
        if idx >= len(dataset):
            continue
            
        print(f"\n样本 {idx} (Episode {idx//320}):")
        sample = dataset[idx]
        
        # 分析状态数据
        if "observation.state" in sample:
            state = sample["observation.state"].numpy()
            print(f"  状态数据: shape={state.shape}, mean={state.mean():.4f}, std={state.std():.4f}")
            print(f"  状态范围: [{state.min():.4f}, {state.max():.4f}]")
            print(f"  前5维: {state[:5]}")
        
        # 分析视频数据
        video_keys = [k for k in sample.keys() if k.startswith("observation.images")]
        print(f"  视频数据: {len(video_keys)} 个摄像头")
        
        for video_key in video_keys:
            video = sample[video_key]
            print(f"    {video_key}: shape={video.shape}, dtype={video.dtype}")
            print(f"      像素范围: [{video.min():.3f}, {video.max():.3f}]")
            print(f"      像素均值: {video.mean():.3f}")
    
    # 检查数据变化性
    print(f"\n🔍 检查数据变化性")
    print("-" * 60)
    
    states = []
    videos = {key: [] for key in ["observation.images.cam_left_high", 
                                  "observation.images.cam_right_high",
                                  "observation.images.cam_left_wrist", 
                                  "observation.images.cam_right_wrist"]}
    
    # 收集多个样本的数据
    for idx in [0, 100, 320, 420, 642, 742]:
        if idx >= len(dataset):
            continue
        sample = dataset[idx]
        
        if "observation.state" in sample:
            states.append(sample["observation.state"].numpy())
        
        for video_key in videos.keys():
            if video_key in sample:
                # 计算视频帧的均值作为特征
                video_mean = sample[video_key].float().mean().item()
                videos[video_key].append(video_mean)
    
    # 分析状态数据变化性
    if states:
        states = np.array(states)
        print(f"状态数据变化性:")
        print(f"  样本间均值差异: {states.std(axis=0).mean():.6f}")
        print(f"  最大变化维度: {states.std(axis=0).max():.6f}")
        print(f"  最小变化维度: {states.std(axis=0).min():.6f}")
    
    # 分析视频数据变化性
    print(f"视频数据变化性:")
    for video_key, values in videos.items():
        if values:
            values = np.array(values)
            print(f"  {video_key}: std={values.std():.6f}, range=[{values.min():.3f}, {values.max():.3f}]")
    
    # 加载模型并测试输入处理
    print(f"\n🔍 测试模型输入处理")
    print("-" * 60)
    
    policy = make_policy(
        hydra_cfg_path="/home/lin/youkechuiguo/youkechuiguo_robot/config/eval/eval_g1_dataset.yaml",
        pretrained_policy_name_or_path="/home/lin/youkechuiguo/model/smolvla_trained/last",
        dataset_repo_id="/home/lin/youkechuiguo/dataset/G1_ObjectPlacement_Dataset"
    )
    policy.eval()
    
    # 测试不同样本的模型输入
    print("测试模型对不同输入的响应:")
    
    predictions = []
    input_features = []
    
    for i, idx in enumerate([0, 320, 642]):
        if idx >= len(dataset):
            continue
            
        sample = dataset[idx]
        
        # 准备模型输入
        batch = {}
        for key, value in sample.items():
            if key.startswith("observation"):
                batch[key] = value.unsqueeze(0).cuda()
        
        batch["task"] = "object placement"
        if hasattr(policy, 'language_tokenizer'):
            batch["language_instruction"] = ["place the object"]
        
        # 获取预测
        with torch.no_grad():
            pred = policy.select_action(batch).cpu().numpy().flatten()
            predictions.append(pred)
        
        # 提取输入特征用于分析
        features = []
        if "observation.state" in sample:
            features.extend(sample["observation.state"].numpy()[:5])  # 前5维状态
        
        # 添加视频特征（每个摄像头的均值）
        for video_key in ["observation.images.cam_left_high", "observation.images.cam_right_high"]:
            if video_key in sample:
                features.append(sample[video_key].float().mean().item())
        
        input_features.append(features)
        
        print(f"  样本 {idx}: pred[0:3]={pred[:3]:.4f}")
    
    predictions = np.array(predictions)
    input_features = np.array(input_features)
    
    print(f"\n📊 输入-输出分析:")
    print(f"输入特征变化性: {input_features.std(axis=0).mean():.6f}")
    print(f"输出预测变化性: {predictions.std(axis=0).mean():.6f}")
    
    # 检查输入特征与输出的相关性
    if len(input_features) > 1 and len(predictions) > 1:
        input_var = input_features.std(axis=0)
        output_var = predictions.std(axis=0)
        
        print(f"输入变化性范围: [{input_var.min():.6f}, {input_var.max():.6f}]")
        print(f"输出变化性范围: [{output_var.min():.6f}, {output_var.max():.6f}]")
    
    # 测试模型内部处理
    print(f"\n🔍 测试模型内部处理")
    print("-" * 60)
    
    # 检查模型是否正确处理视频输入
    sample = dataset[642]
    batch = {}
    for key, value in sample.items():
        if key.startswith("observation"):
            batch[key] = value.unsqueeze(0).cuda()
    
    batch["task"] = "object placement"
    if hasattr(policy, 'language_tokenizer'):
        batch["language_instruction"] = ["place the object"]
    
    # 检查模型内部的normalize_inputs
    print("检查输入标准化:")
    if hasattr(policy, 'normalize_inputs'):
        normalized_batch = policy.normalize_inputs(batch)
        
        for key in batch.keys():
            if key.startswith("observation"):
                original = batch[key]
                normalized = normalized_batch[key]
                
                if original.dtype == torch.float32:  # 只检查数值数据
                    print(f"  {key}:")
                    print(f"    原始: mean={original.mean():.4f}, std={original.std():.4f}")
                    print(f"    标准化: mean={normalized.mean():.4f}, std={normalized.std():.4f}")
    
    # 最终诊断
    print(f"\n" + "=" * 80)
    print(f"🏥 多模态数据诊断")
    print(f"=" * 80)
    
    input_variation = input_features.std(axis=0).mean() if len(input_features) > 1 else 0
    output_variation = predictions.std(axis=0).mean() if len(predictions) > 1 else 0
    
    if input_variation > 0.01 and output_variation < 0.01:
        print(f"❌ 发现问题: 输入数据有变化 ({input_variation:.6f}) 但输出几乎不变 ({output_variation:.6f})")
        print(f"   可能原因:")
        print(f"   1. 模型没有学会处理视觉输入")
        print(f"   2. 视觉编码器权重未正确加载")
        print(f"   3. 输入预处理有问题")
        print(f"   4. 模型过度依赖某种输入模态")
    elif input_variation < 0.01:
        print(f"❌ 发现问题: 输入数据本身变化很小 ({input_variation:.6f})")
        print(f"   可能原因:")
        print(f"   1. 数据集质量问题 - 场景过于相似")
        print(f"   2. 数据预处理过度标准化")
        print(f"   3. 视频数据损坏或格式问题")
    else:
        print(f"✅ 输入输出变化性正常")
        print(f"   输入变化性: {input_variation:.6f}")
        print(f"   输出变化性: {output_variation:.6f}")

if __name__ == "__main__":
    analyze_multimodal_data()
