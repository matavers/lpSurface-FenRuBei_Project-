"""
模型测试脚本

用于在验证集上测试训练好的模型，评估拟合情况。
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.neuralDevelopableSurfaceFitter import NeuralDevelopableSurfaceFitter
from train import DevelopableSurfaceDataset, custom_collate_fn, chamfer_distance, generate_surface_from_control_points
from torch.utils.data import DataLoader


def test_model(model_path, test_data_path, batch_size=32):
    """
    测试模型
    Args:
        model_path: 模型权重文件路径
        test_data_path: 测试数据路径
        batch_size: 批次大小
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = NeuralDevelopableSurfaceFitter(M=16).to(device)
    
    # 加载模型权重
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"模型已从 {model_path} 加载")
    else:
        print(f"警告：模型文件 {model_path} 不存在")
        return
    
    # 创建数据加载器
    test_dataset = DevelopableSurfaceDataset(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    
    # 测试模型
    model.eval()
    total_loss = 0
    total_data_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            interior_points = batch['interior_points'].to(device)
            edge_points = batch['edge_points'].to(device)
            partition_type = batch['partition_type'].to(device)
            target_control_points = batch['control_points'].to(device)
            
            # 前向传播
            pred_control_points = model(interior_points, edge_points, partition_type)
            
            # 生成直纹面点云
            pred_surface = generate_surface_from_control_points(pred_control_points, partition_type, model.M)
            target_surface = generate_surface_from_control_points(target_control_points, partition_type, model.M)
            
            # 计算损失
            data_loss = chamfer_distance(pred_surface, target_surface)
            total_loss += data_loss.item()
            total_data_loss += data_loss.item()
            num_batches += 1
    
    # 计算平均损失
    avg_loss = total_loss / num_batches
    avg_data_loss = total_data_loss / num_batches
    
    print(f"\n测试结果:")
    print(f"平均验证损失: {avg_loss:.6f}")
    print(f"平均数据损失: {avg_data_loss:.6f}")
    
    return avg_loss


def visualize_results(model, test_loader, num_samples=5, training_epochs=100):
    """
    可视化模型预测结果
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        num_samples: 可视化样本数量
        training_epochs: 训练次数
    """
    # 创建输出目录
    result_dir = f"result/{training_epochs}"
    os.makedirs(result_dir, exist_ok=True)
    
    model.eval()
    device = next(model.parameters()).device
    sample_count = 0
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if sample_count >= num_samples:
                break
            
            interior_points = batch['interior_points'].to(device)
            edge_points = batch['edge_points'].to(device)
            partition_type = batch['partition_type'].to(device)
            target_control_points = batch['control_points'].to(device)
            
            # 前向传播
            pred_control_points = model(interior_points, edge_points, partition_type)
            
            # 生成直纹面点云
            pred_surface = generate_surface_from_control_points(pred_control_points, partition_type, model.M)
            target_surface = generate_surface_from_control_points(target_control_points, partition_type, model.M)
            
            # 计算每个样本的损失
            for j in range(interior_points.shape[0]):
                if sample_count >= num_samples:
                    break
                sample_pred = pred_surface[j].unsqueeze(0)
                sample_target = target_surface[j].unsqueeze(0)
                loss = chamfer_distance(sample_pred, sample_target)
                sample_count += 1
                
                print(f"样本 {sample_count} 损失: {loss.item():.6f}")
                
                # 保存点云用于可视化
                np.save(f"{result_dir}/pred_surface_{sample_count}.npy", pred_surface[j].cpu().numpy())
                np.save(f"{result_dir}/target_surface_{sample_count}.npy", target_surface[j].cpu().numpy())
                print(f"点云已保存为 {result_dir}/pred_surface_{sample_count}.npy 和 {result_dir}/target_surface_{sample_count}.npy")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试直纹面拟合神经网络')
    parser.add_argument('--model_path', type=str, default='data/neural/checkpoints/best_model.pth',
                        help='模型权重文件路径')
    parser.add_argument('--test_data', type=str, default='dataset/test/test_dataset.npy',
                        help='测试数据路径')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化结果')
    parser.add_argument('--training_epochs', type=int, default=100,
                        help='训练次数')
    
    args = parser.parse_args()
    
    # 测试模型
    test_model(args.model_path, args.test_data, args.batch_size)
    
    # 可视化结果（如果需要）
    if args.visualize:
        # 创建模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = NeuralDevelopableSurfaceFitter(M=16).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        # 创建数据加载器
        test_dataset = DevelopableSurfaceDataset(args.test_data)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
        
        # 可视化结果
        visualize_results(model, test_loader, training_epochs=args.training_epochs)
