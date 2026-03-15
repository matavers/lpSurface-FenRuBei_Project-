"""
EdgePointToNURBSurfaceNet 训练脚本

本脚本用于训练从边缘点列和内部点云预测NURBS直纹面参数的神经网络。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys
import argparse
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 导入模型
from core.edgePointToNURBSSurfaceNet import EdgePointToNURBSSurfaceNet


class NURBSSurfaceDataset(Dataset):
    """
    NURBS直纹面训练数据集
    """
    
    def __init__(self, data_path):
        """
        初始化数据集
        Args:
            data_path: 数据文件路径
        """
        self.data = np.load(data_path, allow_pickle=True)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取数据项
        Args:
            idx: 数据索引
        Returns:
            边缘点列、内部点云、NURBS参数
        """
        item = self.data[idx]
        
        # 提取数据
        edges = item['edges']  # (4, 64, 3)
        point_cloud = item['point_cloud']  # (1000, 3)
        nurbs_params = item['nurbs_params']  # 包含两条曲线的参数
        
        # 转换为张量
        edges = torch.tensor(edges, dtype=torch.float32)
        point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
        nurbs_params = torch.tensor(nurbs_params, dtype=torch.float32)
        
        return edges, point_cloud, nurbs_params


def train(model, train_loader, val_loader, epochs, learning_rate, device, checkpoint_dir):
    """
    训练模型
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        learning_rate: 学习率
        device: 设备
        checkpoint_dir: 检查点保存目录
    """
    # 定义损失函数
    criterion = nn.MSELoss()
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    
    # 最佳验证损失
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # 训练模式
        model.train()
        train_loss = 0.0
        
        for batch_idx, (edges, point_cloud, targets) in enumerate(train_loader):
            # 移动数据到设备
            edges = edges.to(device)
            point_cloud = point_cloud.to(device)
            targets = targets.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(edges, point_cloud)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 累积损失
            train_loss += loss.item()
            
            # 打印进度
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证模式
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for edges, point_cloud, targets in val_loader:
                # 移动数据到设备
                edges = edges.to(device)
                point_cloud = point_cloud.to(device)
                targets = targets.to(device)
                
                # 前向传播
                outputs = model(edges, point_cloud)
                
                # 计算损失
                loss = criterion(outputs, targets)
                
                # 累积损失
                val_loss += loss.item()
        
        # 计算平均验证损失
        avg_val_loss = val_loss / len(val_loader)
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        # 打印 epoch 结果
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, 'best_nurbs_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'最佳模型已保存到: {checkpoint_path}')
    
    print(f'训练完成！最佳验证损失: {best_val_loss:.4f}')


def main():
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练 EdgePointToNURBSSurfaceNet')
    parser.add_argument('--train_data', type=str, required=True, help='训练数据路径')
    parser.add_argument('--val_data', type=str, required=True, help='验证数据路径')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='检查点保存目录')
    parser.add_argument('--M', type=int, default=16, help='每条曲线的控制点数量')
    parser.add_argument('--degree', type=int, default=3, help='NURBS曲线次数')
    
    args = parser.parse_args()
    
    # 创建检查点目录
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    
    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建数据集
    train_dataset = NURBSSurfaceDataset(args.train_data)
    val_dataset = NURBSSurfaceDataset(args.val_data)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    model = EdgePointToNURBSSurfaceNet(M=args.M, degree=args.degree).to(device)
    print(f'模型参数数量: {sum(p.numel() for p in model.parameters())}')
    
    # 开始训练
    train(model, train_loader, val_loader, args.epochs, args.lr, device, args.checkpoint_dir)


if __name__ == '__main__':
    main()
