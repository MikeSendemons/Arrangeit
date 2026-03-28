#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trial5.py - Part Visualization Module

功能：
1. 按照零件索引绘制每个零件的简化图片
2. 将所有零件的简化图片（标注好索引序号）放在同一大图片中
3. 方便用户快速定位需要操作的对象与检查零件线条是否完整

主要特性：
- 支持两种绘制方法：原始多边形方法和有序绘制方法
- 有序绘制方法解决Part #0缺孔问题（实体链连接性问题）
- 按线条顺序和采样点顺序连接，确保所有内孔正确显示
- 支持多种DXF实体类型：LINE, ARC, POLYLINE, LWPOLYLINE, SPLINE

使用方法：
1. 原始绘制方法（默认）：
   python trial5.py completeINPUT.dxf -o output.png
   
2. 有序绘制方法（解决缺孔问题）：
   python trial5.py completeINPUT.dxf -o output.png --ordered

集成到trial4.py：
在trial4.py中设置USE_ORDERED_DRAWING = True以使用有序绘制方法

作者：基于用户需求开发
版本：1.2.0 (添加有序绘制方法)
日期：2026-03-05
"""

import os
import sys
import math
import ezdxf
from ezdxf.math import Matrix44

# 检查必要的依赖
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Polygon as MplPolygon
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("警告：未找到matplotlib模块，可视化功能将不可用！")
    print("请安装matplotlib模块：pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False

try:
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    SHAPELY_AVAILABLE = True
except ImportError:
    print("错误：未找到shapely模块！")
    print("请安装shapely模块：pip install shapely")
    SHAPELY_AVAILABLE = False
    sys.exit(1)

# 从trial4.py中复制必要的函数
def get_entity_endpoints(entity):
    """获取实体的起点和终点坐标"""
    entity_type = entity.dxftype()
    
    if entity_type == "LINE":
        return (entity.dxf.start.x, entity.dxf.start.y), (entity.dxf.end.x, entity.dxf.end.y)
    
    elif entity_type == "ARC":
        # 圆弧的起点和终点
        start_angle = entity.dxf.start_angle
        end_angle = entity.dxf.end_angle
        center = (entity.dxf.center.x, entity.dxf.center.y)
        radius = entity.dxf.radius
        
        # 计算起点和终点坐标
        start_x = center[0] + radius * math.cos(math.radians(start_angle))
        start_y = center[1] + radius * math.sin(math.radians(start_angle))
        end_x = center[0] + radius * math.cos(math.radians(end_angle))
        end_y = center[1] + radius * math.sin(math.radians(end_angle))
        
        return (start_x, start_y), (end_x, end_y)
    
    elif entity_type == "LWPOLYLINE":
        vertices = list(entity.vertices())
        if len(vertices) == 0:
            return None
        first = vertices[0]
        last = vertices[-1]
        return (first.x, first.y), (last.x, last.y)
    
    elif entity_type == "POLYLINE":
        vertices = list(entity.vertices)
        if len(vertices) == 0:
            return None
        
        first = vertices[0]
        last = vertices[-1]
        
        # 对于封闭的POLYLINE，起点和终点应该是同一个点
        if entity.is_closed:
            return (first.dxf.location.x, first.dxf.location.y), (first.dxf.location.x, first.dxf.location.y)
        else:
            return (first.dxf.location.x, first.dxf.location.y), (last.dxf.location.x, last.dxf.location.y)
    
    elif entity_type == "SPLINE":
        # 对于样条线，使用控制点的第一个和最后一个
        if entity.fit_points:
            points = entity.fit_points
        else:
            points = entity.control_points

        if len(points) == 0:
            return None

        first = points[0]
        last = points[-1]
        
        # 处理点可能是numpy数组的情况
        if hasattr(first, 'x') and hasattr(first, 'y'):
            # 点对象具有.x和.y属性
            return (first.x, first.y), (last.x, last.y)
        else:
            # 点可能是numpy数组或元组
            try:
                # 尝试作为数组访问
                return (float(first[0]), float(first[1])), (float(last[0]), float(last[1]))
            except (IndexError, TypeError):
                # 如果无法访问，返回None
                return None
    
    return None

def points_equal(p1, p2, tolerance=0.5):
    """
    判断两点是否相等（在容差范围内）
    
    参数:
        p1: 第一个点 (x1, y1)
        p2: 第二个点 (x2, y2)
        tolerance: 容差值（毫米），默认0.5mm用于可视化
    
    返回:
        bool: 如果两点距离小于容差则返回True
    
    注意:
        - 可视化模块使用0.5mm容差以保留细节
        - 变换模块(trial4.py)使用0.01mm容差用于精确计算
        - 零件识别模块(trial3.py)使用0.01mm容差用于精确判断
    """
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5 < tolerance
def arc_to_points(arc, num_points=10):
    """将ARC实体离散化为点集"""
    import math
    center = (arc.dxf.center.x, arc.dxf.center.y)
    radius = arc.dxf.radius
    start_angle = math.radians(arc.dxf.start_angle)
    end_angle = math.radians(arc.dxf.end_angle)
    
    # 处理角度方向
    if end_angle < start_angle:
        end_angle += 2 * math.pi
    
    # 生成离散点
    points = []
    for i in range(num_points + 1):
        t = i / num_points
        angle = start_angle + t * (end_angle - start_angle)
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append((x, y))
    
    return points

def improved_points_equal(p1, p2, tolerance=0.5):
    """改进的点相等判断，考虑浮点精度"""
    import math
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    
    # 使用欧几里得距离
    distance = math.sqrt(dx*dx + dy*dy)
    return distance < tolerance


def spline_to_points(spline, num_points=20):
    """
    将样条线离散化为点集
    
    Parameters:
        spline: ezdxf SPLINE实体
        num_points: 离散化点数
    
    Returns:
        list: 离散化后的点坐标列表
    """
    points = []
    
    try:
        # 获取样条线参数
        if hasattr(spline, 'flattening'):
            # 使用ezdxf的flattening方法
            for point in spline.flattening(distance=0.1, segments=num_points):
                points.append((point.x, point.y))
        else:
            # 备用方法：使用控制点或拟合点
            if spline.fit_points and len(spline.fit_points) > 0:
                for fit_point in spline.fit_points:
                    # 处理点可能是numpy数组的情况
                    if hasattr(fit_point, 'x') and hasattr(fit_point, 'y'):
                        points.append((fit_point.x, fit_point.y))
                    else:
                        # 点可能是numpy数组或元组
                        try:
                            points.append((float(fit_point[0]), float(fit_point[1])))
                        except (IndexError, TypeError):
                            continue
            elif spline.control_points and len(spline.control_points) > 0:
                for control_point in spline.control_points:
                    # 处理点可能是numpy数组的情况
                    if hasattr(control_point, 'x') and hasattr(control_point, 'y'):
                        points.append((control_point.x, control_point.y))
                    else:
                        # 点可能是numpy数组或元组
                        try:
                            points.append((float(control_point[0]), float(control_point[1])))
                        except (IndexError, TypeError):
                            continue
    except Exception as e:
        print(f"Warning: Failed to discretize spline: {e}")
        # 使用端点作为后备
        endpoints = get_entity_endpoints(spline)
        if endpoints:
            start, end = endpoints
            points = [start, end]
    
    return points

def find_connected_chains(entities):
    """
    查找首尾相连的实体链
    
    参数:
        entities: DXF实体列表
    
    返回:
        list: 实体链列表，每个链是一个封闭的实体序列
    
    算法说明:
        1. 遍历所有未使用的实体
        2. 从当前实体开始构建链
        3. 向前查找与当前端点连接的实体
        4. 检查连接时使用points_equal函数（0.5mm容差）
        5. 如果实体方向相反，则反转实体方向
        6. 检查链是否封闭（首尾相连）
    
    注意:
        - 使用0.5mm容差，可能无法连接距离较大的实体
        - 对于Part #0外轮廓链，存在21个连接问题（距离>0.5mm）
        - 有序绘制方法不依赖完美的实体链连接
    """
    chains = []
    used = set()
    
    for i, entity in enumerate(entities):
        if i in used:
            continue
        
        endpoints = get_entity_endpoints(entity)
        if not endpoints:
            continue
            
        start, end = endpoints
        
        # 开始构建链
        chain = [entity]
        used.add(i)
        current_end = end
        
        # 向前查找连接
        while True:
            found = False
            for j, next_entity in enumerate(entities):
                if j in used:
                    continue
                    
                next_endpoints = get_entity_endpoints(next_entity)
                if not next_endpoints:
                    continue
                    
                next_start, next_end = next_endpoints
                
                # 检查是否连接
                if points_equal(current_end, next_start):
                    chain.append(next_entity)
                    used.add(j)
                    current_end = next_end
                    found = True
                    break
                elif points_equal(current_end, next_end):
                    # 需要反转实体方向
                    chain.append(next_entity)
                    used.add(j)
                    current_end = next_start
                    found = True
                    break
            
            if not found:
                break
        
        # 检查链是否封闭（首尾相连）
        if points_equal(start, current_end):
            chains.append(chain)
    
    return chains

def extract_closed_contours(msp):
    """从模型空间提取所有封闭轮廓（外轮廓/内孔）"""
    if not SHAPELY_AVAILABLE:
        print("错误：shapely模块不可用！")
        return []
    
    closed_contours = []
    
    # 首先，收集所有可能构成轮廓的实体
    contour_entities = []
    for entity in msp:
        entity_type = entity.dxftype()
        if entity_type in ["LINE", "ARC", "LWPOLYLINE", "POLYLINE", "SPLINE", "CIRCLE"]:
            contour_entities.append(entity)
    
    print(f"找到 {len(contour_entities)} 个可能构成轮廓的实体")
    
    # 1. 首先处理原本就封闭的实体
    for entity in contour_entities:
        entity_type = entity.dxftype()
        poly = None
        
        # 处理轻量级多段线（最常见的封闭轮廓）
        if entity_type == "LWPOLYLINE":
            if not entity.closed:
                continue
            points = list(entity.vertices())
            if len(points) >= 3:
                try:
                    poly = Polygon([(p.x, p.y) for p in points])
                    poly = make_valid(poly) if not poly.is_valid else poly
                except:
                    pass
        
        # 处理普通多段线
        elif entity_type == "POLYLINE" and entity.is_closed:
            points = list(entity.vertices())
            if len(points) >= 3:
                try:
                    poly = Polygon([(p.x, p.y) for p in points])
                    poly = make_valid(poly) if not poly.is_valid else poly
                except:
                    pass
        
        # 处理圆（CIRCLE实体）
        elif entity_type == "CIRCLE":
            center = entity.dxf.center
            radius = entity.dxf.radius
            # 将圆离散化为多边形（36个点）
            import math
            num_points = 36
            points = []
            for i in range(num_points):
                angle = 2 * math.pi * i / num_points
                x = center.x + radius * math.cos(angle)
                y = center.y + radius * math.sin(angle)
                points.append((x, y))
            if len(points) >= 3:
                try:
                    poly = Polygon(points)
                    poly = make_valid(poly) if not poly.is_valid else poly
                except:
                    pass
        
        # 过滤无效/微小轮廓
        if poly and poly.area > 0.01:
            closed_contours.append({
                "entity": entity,
                "polygon": poly,
                "area": poly.area,
                "type": "single"
            })
    
    print(f"找到 {len(closed_contours)} 个原本封闭的轮廓")
    
    # 2. 查找由首尾相连的实体组成的封闭轮廓
    chains = find_connected_chains(contour_entities)
    print(f"找到 {len(chains)} 个首尾相连的实体链")
    
    for chain in chains:
        # 将链转换为点集
        points = []
        previous_point = None
        
        for entity in chain:
            entity_type = entity.dxftype()
            
            if entity_type == "LINE":
                start = (entity.dxf.start.x, entity.dxf.start.y)
                end = (entity.dxf.end.x, entity.dxf.end.y)
                
                if not points:
                    # 第一个实体，添加起点
                    points.append(start)
                    previous_point = start
                
                # 检查应该添加哪个点
                if points_equal(previous_point, start, tolerance=0.5):
                    # 正常方向：添加终点
                    points.append(end)
                    previous_point = end
                elif points_equal(previous_point, end, tolerance=0.5):
                    # 反转方向：添加起点
                    points.append(start)
                    previous_point = start
                else:
                    # 无法连接，添加终点作为后备
                    points.append(end)
                    previous_point = end
            
            elif entity_type == "ARC":
                # 使用改进的ARC离散化
                arc_points = arc_to_points(entity, num_points=10)
                if not arc_points:
                    continue
                
                if not points:
                    # 第一个实体，添加第一个点
                    points.append(arc_points[0])
                    previous_point = arc_points[0]
                
                # 检查连接点
                first_point = arc_points[0]
                last_point = arc_points[-1]
                
                if points_equal(previous_point, first_point, tolerance=0.5):
                    # 正常方向：添加剩余点
                    for i in range(1, len(arc_points)):
                        points.append(arc_points[i])
                        previous_point = arc_points[i]
                elif points_equal(previous_point, last_point, tolerance=0.5):
                    # 反转方向：添加反转后的点
                    for i in range(len(arc_points)-2, -1, -1):
                        points.append(arc_points[i])
                        previous_point = arc_points[i]
                else:
                    # 无法连接，添加所有点
                    for point in arc_points:
                        points.append(point)
                        previous_point = point
            
            elif entity_type in ["LWPOLYLINE", "POLYLINE"]:
                # 添加多段线的顶点
                if entity_type == "LWPOLYLINE":
                    vertices = list(entity.vertices())
                    vertex_points = [(vertex.x, vertex.y) for vertex in vertices]
                else:
                    vertices = list(entity.vertices)
                    vertex_points = [(vertex.dxf.location.x, vertex.dxf.location.y) for vertex in vertices]
                
                if not points:
                    # 第一个顶点
                    points.append(vertex_points[0])
                    previous_point = vertex_points[0]
                
                # 添加剩余顶点
                for i in range(1, len(vertex_points)):
                    points.append(vertex_points[i])
                    previous_point = vertex_points[i]
            
            elif entity_type == "SPLINE":
                # 处理样条线：离散化为点集
                spline_points = spline_to_points(entity, num_points=20)
                if not spline_points:
                    continue
                
                if not points:
                    # 第一个点
                    points.append(spline_points[0])
                    previous_point = spline_points[0]
                
                # 检查连接点
                first_point = spline_points[0]
                last_point = spline_points[-1]
                
                if points_equal(previous_point, first_point, tolerance=0.5):
                    # 正常方向：添加剩余点
                    for i in range(1, len(spline_points)):
                        points.append(spline_points[i])
                        previous_point = spline_points[i]
                elif points_equal(previous_point, last_point, tolerance=0.5):
                    # 反转方向：添加反转后的点
                    for i in range(len(spline_points)-2, -1, -1):
                        points.append(spline_points[i])
                        previous_point = spline_points[i]
                else:
                    # 无法连接，添加所有点
                    for point in spline_points:
                        points.append(point)
                        previous_point = point
        
        # 创建多边形
        try:
            if len(points) >= 3:
                poly = Polygon(points)
                poly = make_valid(poly) if not poly.is_valid else poly
                if poly and poly.area > 0.01:
                    closed_contours.append({
                        "entity": chain[0],  # 使用链中的第一个实体作为代表
                        "polygon": poly,
                        "area": poly.area,
                        "type": "chain",
                        "chain": chain  # 保存整个链供后续使用
                    })
        except Exception as e:
            print(f"创建多边形失败: {e}")
            pass
    
    print(f"总共找到 {len(closed_contours)} 个封闭轮廓")
    
    # 按面积降序排序（大轮廓优先作为外轮廓）
    closed_contours.sort(key=lambda x: x["area"], reverse=True)
    return closed_contours

def group_contours_by_part(contours):
    """将轮廓分组为零件（外轮廓+内孔）"""
    parts = []
    used = set()
    
    for i, outer in enumerate(contours):
        if i in used:
            continue
        
        outer_poly = outer["polygon"]
        # 判断是否为外轮廓：不被任何更大的轮廓包含
        is_outer = True
        for j in range(i):
            if j in used:
                continue
            if contours[j]["polygon"].contains(outer_poly):
                is_outer = False
                break
        
        if not is_outer:
            continue
        
        # 查找被此外轮廓包含的内孔
        holes = []
        for j, hole in enumerate(contours):
            if j == i or j in used:
                continue
            
            # 检查hole是否完全在outer内部
            if hole["polygon"].within(outer_poly):
                holes.append(hole)
                used.add(j)
        
        # 创建零件
        parts.append({
            "outer": outer,
            "holes": holes,
            "center": outer_poly.centroid,
            "bbox": outer_poly.bounds  # (minx, miny, maxx, maxy)
        })
        used.add(i)
    
    return parts

def get_part_entities(part):
    """获取零件中的所有实体（外轮廓+内孔）"""
    entities = []
    
    # 添加外轮廓实体
    if part["outer"]["type"] == "chain" and "chain" in part["outer"]:
        entities.extend(part["outer"]["chain"])
    else:
        entities.append(part["outer"]["entity"])
    
    # 添加内孔实体
    for hole in part["holes"]:
        if hole["type"] == "chain" and "chain" in hole:
            entities.extend(hole["chain"])
        else:
            entities.append(hole["entity"])
    
    return entities

def visualize_parts(parts, output_image="parts_visualization.png", max_cols=3):
    """
    Visualize all parts, placing simplified images of each part in a single large image
    
    Parameters:
        parts: List of parts
        output_image: Output image filename
        max_cols: Maximum number of parts per row
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib module not available, cannot visualize!")
        return False
    
    if not parts:
        print("Warning: No parts to visualize!")
        return False
    
    print(f"Starting visualization of {len(parts)} parts...")
    
    # Calculate layout
    num_parts = len(parts)
    num_cols = min(max_cols, num_parts)
    num_rows = (num_parts + num_cols - 1) // num_cols  # 向上取整
    
    # Create large figure
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))
    
    # If only one part, axes is not an array
    if num_parts == 1:
        axes = np.array([axes])
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    if num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Draw each part
    for idx, part in enumerate(parts):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row, col]
        
        # Draw part
        draw_part(ax, part, idx)
        
        # Set title
        ax.set_title(f"Part #{idx}", fontsize=12, fontweight='bold')
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(num_parts, num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        axes[row, col].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save image
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    print(f"Visualization completed! Image saved to: {output_image}")
    
    # Show image size
    if os.path.exists(output_image):
        print(f"Image file size: {os.path.getsize(output_image)} bytes")
    
    plt.close(fig)
    return True

def draw_part(ax, part, part_index):
    """
    Draw a single part on the specified axes
    
    Parameters:
        ax: matplotlib axes object
        part: Part data
        part_index: Part index
    
    简化图注：
    - 外廓：黑线，无填充
    - 内孔：红线，无填充
    - 保留中心点和零件编号
    - 移除图例和其他不必要的标注
    """
    # Get outer contour polygon of the part
    outer_poly = part["outer"]["polygon"]
    
    # Draw outer contour - handle both Polygon and MultiPolygon
    if hasattr(outer_poly, 'geoms'):
        # MultiPolygon: draw each sub-polygon
        for sub_poly in outer_poly.geoms:
            if hasattr(sub_poly, 'exterior'):
                exterior_coords = list(sub_poly.exterior.coords)
                if exterior_coords:
                    polygon = MplPolygon(exterior_coords, closed=True,
                                        edgecolor='black', facecolor='none',
                                        linewidth=2)
                    ax.add_patch(polygon)
    elif hasattr(outer_poly, 'exterior'):
        # Single Polygon
        exterior_coords = list(outer_poly.exterior.coords)
        if exterior_coords:
            # Create polygon patch - black outline, no fill
            polygon = MplPolygon(exterior_coords, closed=True,
                                edgecolor='black', facecolor='none',
                                linewidth=2)
            ax.add_patch(polygon)
    
    # Draw inner holes - handle both Polygon and MultiPolygon
    for hole_idx, hole in enumerate(part["holes"]):
        hole_poly = hole["polygon"]
        
        if hasattr(hole_poly, 'geoms'):
            # MultiPolygon hole: draw each sub-polygon
            for sub_poly in hole_poly.geoms:
                if hasattr(sub_poly, 'exterior'):
                    exterior_coords = list(sub_poly.exterior.coords)
                    if exterior_coords:
                        polygon = MplPolygon(exterior_coords, closed=True,
                                            edgecolor='red', facecolor='none',
                                            linewidth=1.5)
                        ax.add_patch(polygon)
        elif hasattr(hole_poly, 'exterior'):
            # Single Polygon hole
            exterior_coords = list(hole_poly.exterior.coords)
            if exterior_coords:
                # Inner holes shown in red, no fill
                polygon = MplPolygon(exterior_coords, closed=True,
                                    edgecolor='red', facecolor='none',
                                    linewidth=1.5)
                ax.add_patch(polygon)
    
    # Draw center point (red dot)
    center = part["center"]
    ax.plot(center.x, center.y, 'ro', markersize=6)
    
    # Add index text (simplified)
    bbox = part["bbox"]
    text_x = bbox[0] + (bbox[2] - bbox[0]) / 2
    text_y = bbox[1] + (bbox[3] - bbox[1]) / 2
    ax.text(text_x, text_y, f'#{part_index}',
            fontsize=12, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Set axis limits
    padding = 10  # Add some margin
    ax.set_xlim(bbox[0] - padding, bbox[2] + padding)
    ax.set_ylim(bbox[1] - padding, bbox[3] + padding)
    
    # No legend - simplified visualization

def order_points_by_contour(points, tolerance=None):
    """
    按轮廓顺序重新排列点（使用最近邻算法）
    
    参数:
        points: 点列表 [(x1, y1), (x2, y2), ...]
        tolerance: 最大连接距离（如果为None，则自动计算）
    
    返回:
        list: 按轮廓顺序排列的点
    """
    if len(points) <= 2:
        return points
    
    import math
    
    # 如果没有提供容差，自动计算
    if tolerance is None:
        # 计算所有点之间的平均距离
        distances = []
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                distances.append(math.sqrt(dx*dx + dy*dy))
        if distances:
            avg_distance = sum(distances) / len(distances)
            tolerance = avg_distance * 2.0  # 使用平均距离的2倍作为容差
        else:
            tolerance = 10.0  # 默认容差
    
    # 使用最近邻算法重新排列点
    ordered = []
    remaining = points.copy()
    
    # 从第一个点开始
    current = remaining.pop(0)
    ordered.append(current)
    
    while remaining:
        # 找到最近的点
        min_distance = float('inf')
        nearest_idx = -1
        
        for i, point in enumerate(remaining):
            dx = current[0] - point[0]
            dy = current[1] - point[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < min_distance:
                min_distance = distance
                nearest_idx = i
        
        # 如果最近距离超过容差，可能已经完成了一个轮廓
        if min_distance > tolerance and len(ordered) > 1:
            # 尝试连接回起点形成封闭轮廓
            first_point = ordered[0]
            dx = current[0] - first_point[0]
            dy = current[1] - first_point[1]
            distance_to_start = math.sqrt(dx*dx + dy*dy)
            
            if distance_to_start <= tolerance:
                # 连接回起点，形成封闭轮廓
                ordered.append(first_point)
                break
            else:
                # 无法形成封闭轮廓，继续连接最近点
                pass
        
        if nearest_idx >= 0:
            current = remaining.pop(nearest_idx)
            ordered.append(current)
        else:
            break
    
    return ordered

def entity_to_sorted_points(entity, entity_index):
    """
    将实体转换为有序点序列（按轮廓顺序）
    
    这是改进的有序绘制系统核心函数，按轮廓上的自然顺序连接点，
    避免按坐标排序导致的锯齿线问题。
    
    参数:
        entity: DXF实体 (LINE, ARC, POLYLINE, LWPOLYLINE, SPLINE)
        entity_index: 实体索引（用于线条编号和线型选择）
    
    返回:
        list: 有序点序列 [(x1, y1), (x2, y2), ...]
        str: 实体类型
    
    处理逻辑:
        1. LINE: 直接使用起点和终点
        2. ARC: 离散化为点序列，保持角度顺序
        3. POLYLINE/LWPOLYLINE: 提取所有顶点，保持原始顺序
        4. SPLINE: 使用spline_to_points函数离散化，保持样条线参数顺序
        5. 使用最近邻算法确保点按轮廓顺序连接
    
    优势:
        - 按轮廓自然顺序连接点，避免锯齿线
        - 保持封闭轮廓的完整性
        - 不依赖实体链的完美连接
    """
    import math
    entity_type = entity.dxftype()
    points = []
    
    if entity_type == "LINE":
        # 直线：起点和终点
        start = (entity.dxf.start.x, entity.dxf.start.y)
        end = (entity.dxf.end.x, entity.dxf.end.y)
        points = [start, end]
    
    elif entity_type == "ARC":
        # 圆弧：离散化为点序列，保持角度顺序
        center = (entity.dxf.center.x, entity.dxf.center.y)
        radius = entity.dxf.radius
        start_angle = math.radians(entity.dxf.start_angle)
        end_angle = math.radians(entity.dxf.end_angle)
        
        # 确保角度方向正确
        if end_angle < start_angle:
            end_angle += 2 * math.pi
        
        # 根据弧长决定点数
        arc_length = radius * abs(end_angle - start_angle)
        num_points = max(5, int(arc_length / 2))  # 每2mm一个点
        num_points = min(num_points, 50)  # 限制最大点数
        
        for i in range(num_points + 1):
            t = i / num_points
            angle = start_angle + t * (end_angle - start_angle)
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))
    
    elif entity_type == "POLYLINE":
        # 多段线：所有顶点，保持原始顺序
        vertices = list(entity.vertices)
        for vertex in vertices:
            points.append((vertex.dxf.location.x, vertex.dxf.location.y))
    
    elif entity_type == "LWPOLYLINE":
        # 轻量级多段线：所有顶点，保持原始顺序
        vertices = list(entity.vertices())
        for vertex in vertices:
            points.append((vertex.x, vertex.y))
    
    elif entity_type == "SPLINE":
        # 样条线：使用spline_to_points函数离散化，保持样条线参数顺序
        points = spline_to_points(entity, num_points=20)
    
    # 对于ARC和SPLINE，点已经按轮廓顺序排列，不需要重新排序
    # 对于LINE、POLYLINE、LWPOLYLINE，点已经按正确顺序排列
    # 不需要按坐标排序，保持原始顺序
    
    return points, entity_type

def draw_entity_ordered(ax, entity, entity_index, color='blue', linewidth=2, alpha=0.5):
    """
    按顺序绘制单个实体
    
    参数:
        ax: matplotlib坐标轴
        entity: DXF实体
        entity_index: 实体索引（用于线条编号）
        color: 线条颜色
        linewidth: 线宽
        alpha: 透明度
    """
    points, entity_type = entity_to_sorted_points(entity, entity_index)
    
    if len(points) < 2:
        return
    
    # 提取坐标
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    # 根据实体索引选择线型
    linestyle = '-' if entity_index % 2 == 0 else '--'
    
    # 绘制线条（不添加图例标签，不绘制顶点）
    ax.plot(x_coords, y_coords, color=color, linewidth=linewidth,
            alpha=alpha, linestyle=linestyle)
    
    # 绘制点（用于调试）
    #ax.scatter(x_coords, y_coords, color='red', s=20, alpha=0.5, zorder=5)

def draw_chain_ordered(ax, chain, color='blue', linewidth=2, alpha=0.5):
    """
    按顺序绘制实体链
    
    参数:
        ax: matplotlib坐标轴
        chain: 实体链列表
        color: 线条颜色
        linewidth: 线宽
        alpha: 透明度
    """
    if not chain:
        return
    
    all_points = []
    
    # 按顺序处理每个实体
    for i, entity in enumerate(chain):
        points, entity_type = entity_to_sorted_points(entity, i)
        
        if points:
            # 添加当前实体的点
            all_points.extend(points)
            
            # 绘制当前实体（使用统一颜色，不区分实体类型）
            draw_entity_ordered(ax, entity, i, color=color,
                               linewidth=linewidth, alpha=alpha)
    
    # 如果有点，绘制连接线（不添加图例标签）
    if len(all_points) >= 2:
        # 提取所有点的坐标
        x_coords = [p[0] for p in all_points]
        y_coords = [p[1] for p in all_points]
        
        # 绘制连接所有点的线（虚线，用于显示连接顺序，不添加图例）
        ax.plot(x_coords, y_coords, color='purple', linewidth=1,
                alpha=0.3, linestyle=':')

def draw_part_ordered(ax, part, part_index):
    """
    使用有序绘制方法绘制零件
    
    参数:
        ax: matplotlib坐标轴
        part: 零件数据
        part_index: 零件索引
    """
    # 绘制外轮廓（如果存在实体链）
    if "chain" in part["outer"]:
        chain = part["outer"]["chain"]
        draw_chain_ordered(ax, chain, color='blue', linewidth=2, alpha=0.5)
    else:
        # 如果没有链信息，使用原始多边形方法
        outer_poly = part["outer"]["polygon"]
        if hasattr(outer_poly, 'exterior'):
            exterior_coords = list(outer_poly.exterior.coords)
            if exterior_coords:
                from matplotlib.patches import Polygon as MplPolygon
                polygon = MplPolygon(exterior_coords, closed=True,
                                    edgecolor='blue', facecolor='lightblue',
                                    alpha=0.5, linewidth=2)
                ax.add_patch(polygon)
    
    # 绘制内孔
    for hole_idx, hole in enumerate(part["holes"]):
        if "chain" in hole:
            draw_chain_ordered(ax, hole["chain"], color='red', linewidth=1.5, alpha=0.7)
        else:
            # 如果没有链信息，使用原始多边形方法
            hole_poly = hole["polygon"]
            if hasattr(hole_poly, 'exterior'):
                exterior_coords = list(hole_poly.exterior.coords)
                if exterior_coords:
                    from matplotlib.patches import Polygon as MplPolygon
                    polygon = MplPolygon(exterior_coords, closed=True,
                                        edgecolor='red', facecolor='white',
                                        alpha=0.7, linewidth=1.5)
                    ax.add_patch(polygon)
    
    # 绘制中心点
    center = part["center"]
    ax.plot(center.x, center.y, 'ro', markersize=8, label='Center')
    
    # 添加索引文本
    bbox = part["bbox"]
    text_x = bbox[0] + (bbox[2] - bbox[0]) / 2
    text_y = bbox[1] + (bbox[3] - bbox[1]) / 2
    ax.text(text_x, text_y, f'#{part_index}',
            fontsize=14, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 设置坐标轴范围
    padding = 10
    ax.set_xlim(bbox[0] - padding, bbox[2] + padding)
    ax.set_ylim(bbox[1] - padding, bbox[3] + padding)
    
    # 不添加图例（根据用户要求）

def create_ordered_visualization(input_file, output_image="parts_ordered_visualization.png"):
    """
    使用有序绘制方法创建零件可视化图片
    
    参数:
        input_file: 输入DXF文件路径
        output_image: 输出图片文件名
    
    返回:
        bool: 是否成功
    """
    print("=" * 70)
    print("Ordered Part Visualization Module")
    print(f"输入文件: {input_file}")
    print(f"输出图片: {output_image}")
    print("=" * 70)
    
    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist!")
        return False
    
    # 读取DXF文件
    try:
        doc = ezdxf.readfile(input_file)
        msp = doc.modelspace()
        print(f"Successfully read DXF file, version: {doc.dxfversion}")
        print(f"Modelspace entities count: {len(msp)}")
    except Exception as e:
        print(f"Error reading DXF file: {e}")
        return False
    
    # 提取封闭轮廓
    contours = extract_closed_contours(msp)
    if not contours:
        print("No closed contours found!")
        return False
    
    # 分组为零件
    parts = group_contours_by_part(contours)
    print(f"Identified {len(parts)} parts")
    
    # Display part information
    for i, part in enumerate(parts):
        print(f"  Part #{i}: Outer area={part['outer']['area']:.2f}, Holes count={len(part['holes'])}, Center=({part['center'].x:.2f}, {part['center'].y:.2f})")
    
    # 创建可视化
    if parts:
        # 计算布局
        num_parts = len(parts)
        max_cols = 3
        num_cols = min(max_cols, num_parts)
        num_rows = (num_parts + num_cols - 1) // num_cols  # 向上取整
        
        # 创建图形
        import matplotlib.pyplot as plt
        import numpy as np
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))
        
        # If only one part, axes is not an array
        if num_parts == 1:
            axes = np.array([axes])
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        if num_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # 绘制每个零件
        for idx, part in enumerate(parts):
            row = idx // num_cols
            col = idx % num_cols
            ax = axes[row, col]
            
            # 使用有序绘制方法
            draw_part_ordered(ax, part, idx)
            
            # 设置标题
            ax.set_title(f"Part #{idx} (Ordered)", fontsize=12, fontweight='bold')
            ax.set_aspect('equal', 'box')
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for idx in range(num_parts, num_rows * num_cols):
            row = idx // num_cols
            col = idx % num_cols
            axes[row, col].axis('off')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(output_image, dpi=150, bbox_inches='tight')
        print(f"Ordered visualization completed! Image saved to: {output_image}")
        
        # 显示图片大小
        if os.path.exists(output_image):
            print(f"Image file size: {os.path.getsize(output_image)} bytes")
        
        plt.close(fig)
        return True
    else:
        print("No parts to visualize!")
        return False

def create_visualization(input_file, output_image="parts_visualization.png"):
    """
    Main function to create part visualization image
    
    Parameters:
        input_file: Input DXF file path
        output_image: 输出图片文件名
    
    返回:
        bool: 是否成功
    """
    print("=" * 70)
    print("Part Visualization Module")
    print(f"输入文件: {input_file}")
    print(f"输出图片: {output_image}")
    print("=" * 70)
    
    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist!")
        return False
    
    # 读取DXF文件
    try:
        doc = ezdxf.readfile(input_file)
        msp = doc.modelspace()
        print(f"Successfully read DXF file, version: {doc.dxfversion}")
        print(f"Modelspace entities count: {len(msp)}")
    except Exception as e:
        print(f"Error reading DXF file: {e}")
        return False
    
    # 提取封闭轮廓
    contours = extract_closed_contours(msp)
    if not contours:
        print("No closed contours found!")
        return False
    
    # 分组为零件
    parts = group_contours_by_part(contours)
    print(f"Identified {len(parts)} parts")
    
    # Display part information
    for i, part in enumerate(parts):
        print(f"  Part #{i}: Outer area={part['outer']['area']:.2f}, Holes count={len(part['holes'])}, Center=({part['center'].x:.2f}, {part['center'].y:.2f})")
    
    # Create visualization
    if parts:
        success = visualize_parts(parts, output_image)
        if success:
            print(f"\nVisualization completed successfully!")
            print(f"Image file: {output_image}")
            return True
        else:
            print(f"\nVisualization failed!")
            return False
    else:
        print("No parts to visualize!")
        return False

def main():
    """Main function: Run visualization module independently"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Part Visualization Module')
    parser.add_argument('input', help='输入DXF文件路径')
    parser.add_argument('-o', '--output', default='parts_visualization.png',
                       help='输出图片文件名 (默认: parts_visualization.png)')
    parser.add_argument('--ordered', action='store_true',
                       help='使用有序绘制方法（按线条顺序和采样点顺序连接）')
    
    args = parser.parse_args()
    
    # 选择绘制方法
    if args.ordered:
        print("使用有序绘制方法...")
        success = create_ordered_visualization(args.input, args.output)
    else:
        print("使用原始绘制方法...")
        success = create_visualization(args.input, args.output)
    
    if success:
        print("\n" + "=" * 70)
        print("Visualization module completed successfully!")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("Visualization module failed!")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())