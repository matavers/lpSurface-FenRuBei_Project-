import numpy as np
import os
import sys
import open3d as o3d

def test_point_cloud_simple():
    """
    简单测试点云数据的加载和基本属性
    """
    print("=== 简单测试点云数据 ===")
    
    # 1. 加载点云
    print("1. 加载点云...")
    ply_file = r"D:\Projects\lpSurface\GM\data\models\FloorChair(1).ply"
    
    if not os.path.exists(ply_file):
        print(f"错误：文件 {ply_file} 不存在")
        return False
    
    point_cloud = o3d.io.read_point_cloud(ply_file)
    print(f"成功加载点云：{len(point_cloud.points)} 个点")
    
    # 2. 检查点云属性
    print("2. 检查点云属性...")
    if not point_cloud.has_points():
        print("错误：点云没有点数据")
        return False
    
    # 计算法线
    print("3. 计算点云法线...")
    point_cloud.estimate_normals()
    if not point_cloud.has_normals():
        print("错误：无法计算点云法线")
        return False
    print("点云法线计算完成")
    
    # 3. 检查点云统计信息
    print("4. 点云统计信息...")
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    
    print(f"点云维度：{points.shape}")
    print(f"法线维度：{normals.shape}")
    print(f"点云边界：")
    print(f"  最小坐标：{np.min(points, axis=0)}")
    print(f"  最大坐标：{np.max(points, axis=0)}")
    print(f"  中心点：{np.mean(points, axis=0)}")
    
    # 4. 保存测试结果
    print("5. 保存测试结果...")
    try:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("output", f"test_point_cloud_simple_{timestamp}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存点云
        o3d.io.write_point_cloud(os.path.join(output_dir, "floor_chair_points.ply"), point_cloud)
        
        # 保存点云数据
        np.save(os.path.join(output_dir, "points.npy"), points)
        np.save(os.path.join(output_dir, "normals.npy"), normals)
        
        # 保存指标
        metrics = {
            'num_points': len(point_cloud.points),
            'point_dimension': points.shape,
            'normal_dimension': normals.shape,
            'min_coordinates': np.min(points, axis=0).tolist(),
            'max_coordinates': np.max(points, axis=0).tolist(),
            'center_point': np.mean(points, axis=0).tolist()
        }
        import json
        with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"测试结果保存到：{output_dir}")
        
    except Exception as e:
        print(f"保存测试结果失败：{e}")
        return False
    
    print("=== 简单测试点云数据完成 ===")
    return True

if __name__ == "__main__":
    success = test_point_cloud_simple()
    if success:
        print("测试成功！点云数据能够正常加载和处理")
    else:
        print("测试失败！点云数据无法正常加载和处理")
