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

sys.path.insert(0, str(Path(__file__).parent.parent))

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
        
    def __len__(self):
        return len(self.data)
    
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
        
        # 获取边缘点列
        edge_points = sample['edge_points']
        padded_edges = []
        for edge in edge_points:
            edge = np.array(edge)
            if len(edge) > self.max_edge_points:
                indices = np.random.choice(len(edge), self.max_edge_points, replace=False)
                edge = edge[indices]
            elif len(edge) < self.max_edge_points:
                padding = np.zeros((self.max_edge_points - len(edge), 3))
                edge = np.vstack([edge, padding])
            padded_edges.append(edge)
        
        # 填充到4条边
        while len(padded_edges) < 4:
            padding = np.zeros((self.max_edge_points, 3))
            padded_edges.append(padding)
        
        # 分区类型
        if sample['partition_type'] == 'triangle':
            partition_type = np.array([1, 0], dtype=np.float32)
        else:
            partition_type = np.array([0, 1], dtype=np.float32)
        
        # 获取目标控制点
        if sample['partition_type'] == 'triangle':
            curve_control = sample['curve_control']
            vertex = sample['vertex']
            # 组合控制点：曲线控制点 + 顶点
            control_points = np.vstack([curve_control, vertex[np.newaxis, :]])
        else:
            curve_A = sample['curve_A_control']
            curve_B = sample['curve_B_control']
            control_points = np.vstack([curve_A, curve_B])
        
        return {
            'interior_points': torch.tensor(interior_points, dtype=torch.float32),
            'edge_points': torch.tensor(np.array(padded_edges), dtype=torch.float32),
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
        control_points: 控制点 (B, 2M+1, 3)
        partition_type: 分区类型 (B, 2)
        M: 控制点数量
        num_u: u方向采样点数
        num_v: v方向采样点数
    Returns:
        直纹面点云 (B, num_u*num_v, 3)
    """
    batch_size = control_points.shape[0]
    is_triangle = partition_type[:, 0] > 0.5
    
    # 生成参数网格
    u = torch.linspace(0, 1, num_u, device=control_points.device)
    v = torch.linspace(0, 1, num_v, device=control_points.device)
    
    # 简单的线性插值（简化版B样条）
    u_expand = u.unsqueeze(1).expand(num_u, num_v)
    v_expand = v.unsqueeze(0).expand(num_u, num_v)
    
    surface_points = []
    
    for b in range(batch_size):
        if is_triangle[b]:
            # 三角形（锥面）
            curve = control_points[b, :M]  # (M, 3)
            vertex = control_points[b, M:]  # (1, 3)
            
            # 插值曲线点
            t = u_expand.flatten()
            curve_idx = t * (M - 1)
            curve_idx_low = curve_idx.long()
            curve_idx_high = torch.clamp(curve_idx_low + 1, 0, M - 1)
            t_local = (curve_idx - curve_idx_low.float()).unsqueeze(-1)
            
            curve_points_low = curve[curve_idx_low]
            curve_points_high = curve[curve_idx_high]
            curve_points = (1 - t_local) * curve_points_low + t_local * curve_points_high
            
            # 生成锥面
            vertex_expand = vertex.unsqueeze(0).expand(num_u * num_v, -1)
            v_expand_flat = v_expand.flatten().unsqueeze(-1)
            
            points = vertex_expand + v_expand_flat * (curve_points - vertex_expand)
        else:
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
    is_triangle = partition_type[:, 0] > 0.5
    
    total_loss = 0
    
    for b in range(batch_size):
        if is_triangle[b]:
            # 三角形：检查曲线端点和顶点
            pred_curve = pred_control_points[b, :M]
            target_curve = target_control_points[b, :M]
            pred_vertex = pred_control_points[b, M:]
            target_vertex = target_control_points[b, M:]
            
            loss = torch.sum((pred_curve[0] - target_curve[0]) ** 2)
            loss += torch.sum((pred_curve[-1] - target_curve[-1]) ** 2)
            loss += torch.sum((pred_vertex - target_vertex) ** 2)
        else:
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
                is_triangle = partition_type[b, 0] > 0.5
                
                if is_triangle:
                    # 三角形：3条边
                    pred_curve = pred_control_points[b, :self.M]
                    pred_vertex = pred_control_points[b, self.M:]
                    
                    # 提取边缘点
                    edge1_pred = pred_surface[b, ::num_u]  # 曲线边
                    edge2_pred = pred_vertex.unsqueeze(0).expand(num_u, -1)  # 顶点到曲线起点
                    edge3_pred = pred_vertex.unsqueeze(0).expand(num_u, -1)  # 顶点到曲线终点
                    
                    target_edge1 = edge_points[b, 0]
                    target_edge2 = edge_points[b, 1]
                    target_edge3 = edge_points[b, 2]
                    
                    edge_loss += chamfer_distance(edge1_pred.unsqueeze(0), target_edge1.unsqueeze(0))
                    edge_loss += chamfer_distance(edge2_pred.unsqueeze(0), target_edge2.unsqueeze(0))
                    edge_loss += chamfer_distance(edge3_pred.unsqueeze(0), target_edge3.unsqueeze(0))
                else:
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


def train_model(train_data_path: str, val_data_path: str, num_epochs: int = 200, 
                batch_size: int = 32, learning_rate: float = 1e-4,
                checkpoint_dir: str = "data/neural/checkpoints"):
    """
    训练模型
    Args:
        train_data_path: 训练数据路径
        val_data_path: 验证数据路径
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        checkpoint_dir: 检查点保存目录
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 创建训练器
    trainer = Trainer(model, device, learning_rate=learning_rate)
    
    # 训练循环
    best_val_loss = float('inf')
    
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
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"最佳模型已保存，验证损失: {best_val_loss:.6f}")
        
        # 定期保存检查点
        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            trainer.save_checkpoint(checkpoint_path, epoch + 1, best_val_loss)
    
    print("\n训练完成！")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练直纹面拟合神经网络')
    parser.add_argument('--train_data', type=str, default='data/neural/train_dataset.npy',
                        help='训练数据路径')
    parser.add_argument('--val_data', type=str, default='data/neural/test_dataset.npy',
                        help='验证数据路径')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--checkpoint_dir', type=str, default='data/neural/checkpoints',
                        help='检查点保存目录')
    
    args = parser.parse_args()
    
    train_model(args.train_data, args.val_data, args.epochs, args.batch_size, args.lr, args.checkpoint_dir)
