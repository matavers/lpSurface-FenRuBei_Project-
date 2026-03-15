"""
神经网络训练脚本

用于训练直纹面拟合神经网络模型。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.neuralDevelopableSurfaceFitter import (
    NeuralDevelopableSurfaceFitter,
    PointNetPPEncoder,
    EdgeEncoder,
    DevelopableSurfaceDecoder
)


class DevelopableSurfaceDataset(Dataset):
    """
    直纹面数据集
    """
    
    def __init__(self, data_path: str, max_interior_points: int = 1000, max_edge_points: int = 32):
        """
        初始化数据集
        Args:
            data_path: 数据文件路径
            max_interior_points: 内部点云最大点数
            max_edge_points: 边缘点最大点数
        """
        self.data = np.load(data_path, allow_pickle=True)
        self.max_interior_points = max_interior_points
        self.max_edge_points = max_edge_points
        # 预处理数据，确保所有样本的边缘点列形状一致
        self._preprocess_data()
        
    def __len__(self):
        return len(self.data)
    
    def _preprocess_data(self):
        """
        预处理数据，确保所有样本的边缘点列和控制点形状一致
        """
        preprocessed_data = []
        for sample in self.data:
            # 初始化边缘点列为固定形状
            processed_sample = sample.copy()
            processed_sample['edge_points'] = []
            
            # 处理原始边缘点列
            original_edges = sample.get('edge_points', [])
            for i in range(4):  # 确保有4条边
                if i < len(original_edges):
                    edge = original_edges[i]
                    edge = np.array(edge)
                    # 确保是2D数组
                    if edge.ndim == 1:
                        edge = edge.reshape(1, 3)
                    # 采样或填充到固定长度
                    if edge.shape[0] > self.max_edge_points:
                        # 随机采样
                        indices = np.random.choice(edge.shape[0], self.max_edge_points, replace=False)
                        edge = edge[indices]
                    elif edge.shape[0] < self.max_edge_points:
                        # 填充
                        padding = np.zeros((self.max_edge_points - edge.shape[0], 3))
                        edge = np.vstack([edge, padding])
                else:
                    # 不足4条边时，创建空边缘点列
                    edge = np.zeros((self.max_edge_points, 3))
                processed_sample['edge_points'].append(edge)
            
            # 处理控制点，确保所有样本的控制点数量一致（32个）
            # 四边形：2*M 控制点（32个）
            curve_A = sample['curve_A_control']
            curve_B = sample['curve_B_control']
            control_points = np.vstack([curve_A, curve_B])
            processed_sample['control_points'] = control_points
            
            preprocessed_data.append(processed_sample)
        
        self.data = preprocessed_data
    
    def __getitem__(self, idx):
        """
        获取一个样本
        """
        sample = self.data[idx]
        
        # 获取内部点云
        interior_points = sample['interior_points']
        
        # 采样或填充到固定数量
        if len(interior_points) > self.max_interior_points:
            indices = np.random.choice(len(interior_points), self.max_interior_points, replace=False)
            interior_points = interior_points[indices]
        elif len(interior_points) < self.max_interior_points:
            padding = np.zeros((self.max_interior_points - len(interior_points), 3))
            interior_points = np.vstack([interior_points, padding])
        
        # 获取边缘点列并转换为numpy数组
        edge_points = sample['edge_points']
        edge_array = np.array(edge_points)
        
        # 分区类型 - 只处理四边形
        partition_type = np.array([0, 1], dtype=np.float32)
        
        # 使用预处理后的控制点
        control_points = sample['control_points']
        
        return {
            'interior_points': torch.tensor(interior_points, dtype=torch.float32),
            'edge_points': torch.tensor(edge_array, dtype=torch.float32),
            'partition_type': torch.tensor(partition_type, dtype=torch.float32),
            'control_points': torch.tensor(control_points, dtype=torch.float32)
        }


def chamfer_distance(pred_points: torch.Tensor, target_points: torch.Tensor):
    """
    计算Chamfer距离
    Args:
        pred_points: 预测点云 (B, N, 3)
        target_points: 目标点云 (B, M, 3)
    Returns:
        Chamfer距离
    """
    # 计算预测点到目标点的最小距离
    pred_points_expand = pred_points.unsqueeze(2)  # (B, N, 1, 3)
    target_points_expand = target_points.unsqueeze(1)  # (B, 1, M, 3)
    
    dist_matrix = torch.sum((pred_points_expand - target_points_expand) ** 2, dim=-1)  # (B, N, M)
    
    min_dist_pred_to_target = torch.min(dist_matrix, dim=2)[0]  # (B, N)
    min_dist_target_to_pred = torch.min(dist_matrix, dim=1)[0]  # (B, M)
    
    chamfer_dist = torch.mean(min_dist_pred_to_target, dim=1) + torch.mean(min_dist_target_to_pred, dim=1)
    
    return torch.mean(chamfer_dist)


def generate_surface_from_control_points(control_points: torch.Tensor, partition_type: torch.Tensor, M: int = 16, 
                                          num_u: int = 32, num_v: int = 32) -> torch.Tensor:
    """
    根据控制点生成直纹面点云
    Args:
        control_points: 控制点 (B, 2M, 3)
        partition_type: 分区类型 (B, 2)
        M: 控制点数量
        num_u: u方向采样点数
        num_v: v方向采样点数
    Returns:
        直纹面点云 (B, num_u*num_v, 3)
    """
    batch_size = control_points.shape[0]
    
    # 生成参数网格
    u = torch.linspace(0, 1, num_u, device=control_points.device)
    v = torch.linspace(0, 1, num_v, device=control_points.device)
    
    # 简单的线性插值（简化版B样条）
    u_expand = u.unsqueeze(1).expand(num_u, num_v)
    v_expand = v.unsqueeze(0).expand(num_u, num_v)
    
    surface_points = []
    
    for b in range(batch_size):
        # 四边形（直纹面）
        curve_A = control_points[b, :M]
        curve_B = control_points[b, M:2*M]
        
        # 插值曲线A点
        t = u_expand.flatten()
        curve_idx = t * (M - 1)
        curve_idx_low = curve_idx.long()
        curve_idx_high = torch.clamp(curve_idx_low + 1, 0, M - 1)
        t_local = (curve_idx - curve_idx_low.float()).unsqueeze(-1)
        
        curve_A_low = curve_A[curve_idx_low]
        curve_A_high = curve_A[curve_idx_high]
        curve_A_points = (1 - t_local) * curve_A_low + t_local * curve_A_high
        
        curve_B_low = curve_B[curve_idx_low]
        curve_B_high = curve_B[curve_idx_high]
        curve_B_points = (1 - t_local) * curve_B_low + t_local * curve_B_high
        
        # 生成直纹面
        v_flat = v_expand.flatten().unsqueeze(-1)
        points = (1 - v_flat) * curve_A_points + v_flat * curve_B_points
        
        surface_points.append(points)
    
    return torch.stack(surface_points)


def smooth_loss(control_points: torch.Tensor, M: int = 16) -> torch.Tensor:
    """
    计算控制点的平滑损失（二阶差分）
    Args:
        control_points: 控制点 (B, N, 3)
        M: 每条曲线的控制点数量
    Returns:
        平滑损失
    """
    batch_size = control_points.shape[0]
    num_curves = control_points.shape[1] // M
    
    total_loss = 0
    
    for c in range(num_curves):
        curve = control_points[:, c*M:(c+1)*M, :]
        
        # 二阶差分
        diff2 = curve[:, 2:, :] - 2 * curve[:, 1:-1, :] + curve[:, :-2, :]
        
        total_loss += torch.mean(diff2 ** 2)
    
    return total_loss / num_curves


def endpoint_loss(pred_control_points: torch.Tensor, target_control_points: torch.Tensor, 
                  partition_type: torch.Tensor, M: int = 16) -> torch.Tensor:
    """
    计算端点约束损失
    """
    batch_size = pred_control_points.shape[0]
    
    total_loss = 0
    
    for b in range(batch_size):
        # 四边形：检查两条曲线的端点
        pred_curve_A = pred_control_points[b, :M]
        pred_curve_B = pred_control_points[b, M:2*M]
        target_curve_A = target_control_points[b, :M]
        target_curve_B = target_control_points[b, M:2*M]
        
        loss = torch.sum((pred_curve_A[0] - target_curve_A[0]) ** 2)
        loss += torch.sum((pred_curve_A[-1] - target_curve_A[-1]) ** 2)
        loss += torch.sum((pred_curve_B[0] - target_curve_B[0]) ** 2)
        loss += torch.sum((pred_curve_B[-1] - target_curve_B[-1]) ** 2)
        
        total_loss += loss
    
    return total_loss / batch_size


class Trainer:
    """
    训练器
    """
    
    def __init__(self, model: nn.Module, device: torch.device, learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5, lambda_data: float = 1.0, lambda_edge: float = 1.0,
                 lambda_smooth: float = 0.1, lambda_end: float = 1.0):
        """
        初始化训练器
        Args:
            model: 神经网络模型
            device: 设备
            learning_rate: 学习率
            weight_decay: 权重衰减
            lambda_data: 数据项损失权重
            lambda_edge: 边缘约束损失权重
            lambda_smooth: 平滑项损失权重
            lambda_end: 端点约束损失权重
        """
        self.model = model
        self.device = device
        self.lambda_data = lambda_data
        self.lambda_edge = lambda_edge
        self.lambda_smooth = lambda_smooth
        self.lambda_end = lambda_end
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        
        self.M = model.M
        
    def train_epoch(self, dataloader: DataLoader) -> dict:
        """
        训练一个epoch
        """
        self.model.train()
        
        total_loss = 0
        total_data_loss = 0
        total_edge_loss = 0
        total_smooth_loss = 0
        total_end_loss = 0
        
        num_batches = 0
        
        for batch in dataloader:
            interior_points = batch['interior_points'].to(self.device)
            edge_points = batch['edge_points'].to(self.device)
            partition_type = batch['partition_type'].to(self.device)
            target_control_points = batch['control_points'].to(self.device)
            
            # 前向传播
            pred_control_points = self.model(interior_points, edge_points, partition_type)
            
            # 生成直纹面点云
            pred_surface = generate_surface_from_control_points(pred_control_points, partition_type, self.M)
            target_surface = generate_surface_from_control_points(target_control_points, partition_type, self.M)
            
            # 计算损失
            data_loss = chamfer_distance(pred_surface, target_surface)
            
            # 边缘约束：计算边界点与边缘点列的Chamfer距离
            edge_loss = torch.tensor(0.0, device=self.device)
            batch_size = interior_points.shape[0]
            num_u = 32
            
            for b in range(batch_size):
                # 四边形：4条边
                pred_curve_A = pred_control_points[b, :self.M]
                pred_curve_B = pred_control_points[b, self.M:2*self.M]
                
                # 提取边缘点
                edge1_pred = pred_surface[b, ::num_u]  # 曲线A (v=0)
                edge2_pred = pred_surface[b, 31::num_u]  # 曲线B (v=1)
                edge3_pred = pred_surface[b, :32]  # u=0
                edge4_pred = pred_surface[b, 32*31:]  # u=1
                
                target_edge1 = edge_points[b, 0]
                target_edge2 = edge_points[b, 1]
                target_edge3 = edge_points[b, 2]
                target_edge4 = edge_points[b, 3]
                
                edge_loss += chamfer_distance(edge1_pred.unsqueeze(0), target_edge1.unsqueeze(0))
                edge_loss += chamfer_distance(edge2_pred.unsqueeze(0), target_edge2.unsqueeze(0))
                edge_loss += chamfer_distance(edge3_pred.unsqueeze(0), target_edge3.unsqueeze(0))
                edge_loss += chamfer_distance(edge4_pred.unsqueeze(0), target_edge4.unsqueeze(0))
            
            edge_loss = edge_loss / batch_size
            
            # 平滑损失
            smooth_loss_value = smooth_loss(pred_control_points, self.M)
            
            # 端点约束损失
            end_loss = endpoint_loss(pred_control_points, target_control_points, partition_type, self.M)
            
            # 总损失
            loss = (self.lambda_data * data_loss + 
                   self.lambda_edge * edge_loss + 
                   self.lambda_smooth * smooth_loss_value + 
                   self.lambda_end * end_loss)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_data_loss += data_loss.item()
            total_edge_loss += edge_loss.item()
            total_smooth_loss += smooth_loss_value.item()
            total_end_loss += end_loss.item()
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'data_loss': total_data_loss / num_batches,
            'edge_loss': total_edge_loss / num_batches,
            'smooth_loss': total_smooth_loss / num_batches,
            'end_loss': total_end_loss / num_batches
        }
    
    def validate(self, dataloader: DataLoader) -> dict:
        """
        验证
        """
        self.model.eval()
        
        total_loss = 0
        total_data_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                interior_points = batch['interior_points'].to(self.device)
                edge_points = batch['edge_points'].to(self.device)
                partition_type = batch['partition_type'].to(self.device)
                target_control_points = batch['control_points'].to(self.device)
                
                # 前向传播
                pred_control_points = self.model(interior_points, edge_points, partition_type)
                
                # 生成直纹面点云
                pred_surface = generate_surface_from_control_points(pred_control_points, partition_type, self.M)
                target_surface = generate_surface_from_control_points(target_control_points, partition_type, self.M)
                
                # 计算损失
                data_loss = chamfer_distance(pred_surface, target_surface)
                total_loss += data_loss.item()
                total_data_loss += data_loss.item()
                num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_data_loss': total_data_loss / num_batches
        }
    
    def save_checkpoint(self, filepath: str, epoch: int, best_loss: float):
        """
        保存检查点
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': best_loss
        }
        torch.save(checkpoint, filepath)
        print(f"检查点已保存到 {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        加载检查点
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"检查点已从 {filepath} 加载")


def custom_collate_fn(batch):
    """
    自定义批次处理函数，确保所有样本的边缘点列形状一致
    """
    # 提取批次中的所有元素
    interior_points = []
    edge_points = []
    partition_types = []
    control_points = []
    
    for item in batch:
        interior_points.append(item['interior_points'])
        edge_points.append(item['edge_points'])
        partition_types.append(item['partition_type'])
        control_points.append(item['control_points'])
    
    # 堆叠内部点云
    interior_points = torch.stack(interior_points, 0)
    
    # 堆叠边缘点列
    edge_points = torch.stack(edge_points, 0)
    
    # 堆叠分区类型
    partition_types = torch.stack(partition_types, 0)
    
    # 堆叠控制点
    control_points = torch.stack(control_points, 0)
    
    return {
        'interior_points': interior_points,
        'edge_points': edge_points,
        'partition_type': partition_types,
        'control_points': control_points
    }


def train_model(train_data_path: str, val_data_path: str, num_epochs: int = 100, 
                batch_size: int = 32, learning_rate: float = 1e-4,
                checkpoint_dir: str = "data/neural/checkpoints",
                patience: int = 10, min_delta: float = 0.001):
    """
    训练模型
    Args:
        train_data_path: 训练数据路径
        val_data_path: 验证数据路径
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        checkpoint_dir: 检查点保存目录
        patience: 早停耐心值，连续多少轮验证损失无改善则停止
        min_delta: 最小改善阈值，小于此值视为无改善
    """
    # 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = NeuralDevelopableSurfaceFitter(M=16).to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 创建数据加载器
    train_dataset = DevelopableSurfaceDataset(train_data_path)
    val_dataset = DevelopableSurfaceDataset(val_data_path)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    
    # 创建训练器
    trainer = Trainer(model, device, learning_rate=learning_rate)
    
    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # 训练
        train_metrics = trainer.train_epoch(train_loader)
        print(f"训练损失: {train_metrics['loss']:.6f} | "
              f"数据: {train_metrics['data_loss']:.6f} | "
              f"边缘: {train_metrics['edge_loss']:.6f} | "
              f"平滑: {train_metrics['smooth_loss']:.6f} | "
              f"端点: {train_metrics['end_loss']:.6f}")
        
        # 验证
        val_metrics = trainer.validate(val_loader)
        print(f"验证损失: {val_metrics['val_loss']:.6f}")
        
        # 更新学习率
        trainer.scheduler.step()
        
        # 保存最佳模型
        if val_metrics['val_loss'] < best_val_loss - min_delta:
            best_val_loss = val_metrics['val_loss']
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"最佳模型已保存，验证损失: {best_val_loss:.6f}")
            patience_counter = 0  # 重置耐心计数器
        else:
            patience_counter += 1
            print(f"早停计数器: {patience_counter}/{patience}")
            
        # 检查是否早停
        if patience_counter >= patience:
            print(f"验证损失在 {patience} 轮内无改善，停止训练")
            break
        
        # 定期保存检查点
        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            trainer.save_checkpoint(checkpoint_path, epoch + 1, best_val_loss)
    
    print("\n训练完成！")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练直纹面拟合神经网络')
    parser.add_argument('--train_data', type=str, default='dataset/train/train_dataset.npy',
                        help='训练数据路径')
    parser.add_argument('--val_data', type=str, default='dataset/test/test_dataset.npy',
                        help='验证数据路径')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--checkpoint_dir', type=str, default='data/neural/checkpoints',
                        help='检查点保存目录')
    parser.add_argument('--patience', type=int, default=10,
                        help='早停耐心值')
    parser.add_argument('--min_delta', type=float, default=0.001,
                        help='早停最小改善阈值')
    parser.add_argument('--visualize_data', action='store_true',
                        help='是否在训练前可视化训练数据')
    
    args = parser.parse_args()
    
    # 跳过可视化，直接开始训练
    print("跳过可视化，直接开始训练...")
    
    # 开始训练
    print("开始训练...")
    train_model(args.train_data, args.val_data, args.epochs, args.batch_size, args.lr, 
                args.checkpoint_dir, args.patience, args.min_delta)
