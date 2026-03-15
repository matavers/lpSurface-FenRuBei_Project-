"""
训练过程可视化工具

实时可视化训练结果，对比预测曲面和目标曲面。
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.edgeToSurfaceNet import EdgeToSurfaceNet


def generate_surface_points(control_points: np.ndarray, M: int = 16, 
                            num_u: int = 32, num_v: int = 32) -> np.ndarray:
    """
    根据控制点生成直纹面点云
    Args:
        control_points: 控制点 (2M, 3)
        M: 每条曲线的控制点数量
        num_u: u 方向采样点数
        num_v: v 方向采样点数
    Returns:
        直纹面点云 (num_u*num_v, 3)
    """
    curve_A = control_points[:M]
    curve_B = control_points[M:2*M]
    
    surface_points = []
    
    for u in np.linspace(0, 1, num_u):
        # 评估曲线 A 上的点（线性插值）
        t = u * (M - 1)
        k = int(np.floor(t))
        if k >= M - 1:
            k = M - 2
        t_local = t - k
        
        if k == 0:
            p0, p1, p2 = curve_A[0], curve_A[1], curve_A[2]
        elif k == M - 2:
            p0, p1, p2 = curve_A[-3], curve_A[-2], curve_A[-1]
        else:
            p0, p1, p2 = curve_A[k], curve_A[k+1], curve_A[k+2]
        
        # 二次 B 样条
        curve_A_point = (1 - t_local)**2 / 2 * p0 + \
                       (1 - 2*t_local + t_local**2) * p1 + \
                       t_local**2 / 2 * p2
        
        # 评估曲线 B 上的点
        if k == 0:
            p0, p1, p2 = curve_B[0], curve_B[1], curve_B[2]
        elif k == M - 2:
            p0, p1, p2 = curve_B[-3], curve_B[-2], curve_B[-1]
        else:
            p0, p1, p2 = curve_B[k], curve_B[k+1], curve_B[k+2]
        
        curve_B_point = (1 - t_local)**2 / 2 * p0 + \
                       (1 - 2*t_local + t_local**2) * p1 + \
                       t_local**2 / 2 * p2
        
        # 沿 v 方向插值生成直纹面
        for v in np.linspace(0, 1, num_v):
            point = (1 - v) * curve_A_point + v * curve_B_point
            surface_points.append(point)
    
    return np.array(surface_points)


def plot_surface_comparison(pred_control: np.ndarray, target_control: np.ndarray, 
                           epoch: int, loss: float, save_path: str = None):
    """
    可视化预测曲面和目标曲面对比
    """
    fig = plt.figure(figsize=(14, 6))
    
    # 生成曲面点
    pred_points = generate_surface_points(pred_control, M=16)
    target_points = generate_surface_points(target_control, M=16)
    
    # 绘制预测曲面
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
               c='red', s=1, alpha=0.6, label='Predicted')
    ax1.set_title(f'Predicted Surface (Epoch {epoch})\nLoss: {loss:.6f}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # 绘制目标曲面
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], 
               c='blue', s=1, alpha=0.6, label='Target')
    ax2.set_title('Target Surface')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curve(metrics_history: list, save_path: str = None):
    """
    绘制训练曲线
    """
    epochs = range(1, len(metrics_history) + 1)
    train_losses = [m['loss'] for m in metrics_history]
    val_losses = [m.get('val_loss', None) for m in metrics_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if val_losses[0] is not None:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"训练曲线已保存到 {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_batch_predictions(model: EdgeToSurfaceNet, dataloader: torch.utils.data.DataLoader,
                                device: torch.device, epoch: int, save_dir: str = "result/visualization"):
    """
    可视化一批次的预测结果
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        batch = next(iter(dataloader))
        edge_points = batch['edge_points'].to(device)
        target_control = batch['control_points'].to(device)
        
        # 预测
        pred_control = model(edge_points)
        
        # 可视化前 4 个样本
        for i in range(min(4, edge_points.shape[0])):
            pred_np = pred_control[i].cpu().numpy()
            target_np = target_control[i].cpu().numpy()
            
            # 计算损失
            pred_points = generate_surface_points(pred_np, M=16)
            target_points = generate_surface_points(target_np, M=16)
            
            # 简单 Chamfer 距离近似
            dist = np.mean(np.sqrt(np.sum((pred_points - target_points) ** 2, axis=1)))
            
            save_path = os.path.join(save_dir, f"epoch_{epoch}_sample_{i}.png")
            plot_surface_comparison(pred_np, target_np, epoch, dist, save_path)
    
    print(f"批次可视化完成，保存到 {save_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化训练结果')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型权重文件路径')
    parser.add_argument('--test_data', type=str, default='dataset/test/test_dataset.npy',
                        help='测试数据路径')
    parser.add_argument('--epoch', type=int, default=0,
                        help='当前训练轮数')
    parser.add_argument('--output_dir', type=str, default='result/visualization',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EdgeToSurfaceNet(M=16).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # 加载数据
    from train_edge_to_surface import EdgeToSurfaceDataset
    dataset = EdgeToSurfaceDataset(args.test_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    
    # 可视化
    visualize_batch_predictions(model, dataloader, device, args.epoch, args.output_dir)
    
    print("可视化完成！")
