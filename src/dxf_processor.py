#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dxf_processor.py - DXF文件处理模块

功能：
1. 识别并分别存储每个零件的所有图线（外廓+内孔）到block中
2. 输出一张所有零件及对应编号的图片，供用户确认零件识别结果，以及在写入input_data编辑各零件数目与可旋转角度时作参考
3. 另外输出一张只有每个零件的外轮廓的图片（验证轮廓提取正确性）
4. 输出每个零件按要求偏移后的位图与二值化后的npy数组（可手动勾选输出原零件的位图）
5. 根据输入文件（中间文件placement_info）对block中的零件进行坐标变换（旋转+平移），复制等操作并输出新的DXF文件

主要特性：
- 零件识别算法的核心是封闭曲线占据区域的包含关系判断，使用shapely库进行几何处理
- 支持多种DXF实体类型：LINE, ARC, POLYLINE, LWPOLYLINE, SPLINE, CIRCLE， ELLIPSE等
- 提供了灵活的参数配置，包括样条线离散化步长、面积过滤阈值、位图生成分辨率等

作者：hhggg
版本：1.0.0
日期：2026-03-13
"""

import os
import sys
import math
import ezdxf
import warnings
warnings.filterwarnings("ignore")  # 忽略shapely的非关键警告

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
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    print("警告：未找到PIL（Pillow）模块，位图填充功能可能受限！")
    print("请安装Pillow模块：pip install Pillow")
    PIL_AVAILABLE = False

try:
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    from shapely.validation import make_valid
    SHAPELY_AVAILABLE = True
except ImportError:
    print("错误：未找到shapely模块！")
    print("请安装shapely模块：pip install shapely")
    SHAPELY_AVAILABLE = False
    sys.exit(1)

# -------------------------- 核心配置 --------------------------
# 文件路径配置
DEFAULT_INPUT_FILE = "data/input/completeINPUT.dxf"  # 默认输入文件路径
DEFAULT_OUTPUT_DXF = "data/output/processed_parts.dxf"  # 默认输出DXF文件路径
DEFAULT_OUTPUT_IMAGE_ALL = "data/output/parts_all.png"  # 所有零件图片
DEFAULT_OUTPUT_IMAGE_OUTER = "data/output/parts_outer_only.png"  # 仅外轮廓图片

# 几何处理配置
PART_BLOCK_PREFIX = "PART_"  # 零件块名称前缀
SAMPLING_STEP = 0.1  # 样条线离散化步长（越小越精确，越慢）
AREA_THRESHOLD = 10.0  # 过滤微小无效轮廓的面积阈值（原为0.01，太小导致内孔被识别为零件）

# 位图生成配置 - 现在从main.py导入全局变量
# 注意：这些变量将在运行时从main.py导入，这里只提供默认值作为后备
try:
    from main import PIXEL_RESOLUTION, SUBSTRATE_WIDTH_MM as DEFAULT_WIDTH_MM, SUBSTRATE_HEIGHT_MM as DEFAULT_HEIGHT_MM, CUTTING_GAP_PG
    print(f"[INFO] 从main.py导入全局配置: PIXEL_RESOLUTION={PIXEL_RESOLUTION}, "
          f"DEFAULT_WIDTH_MM={DEFAULT_WIDTH_MM}, DEFAULT_HEIGHT_MM={DEFAULT_HEIGHT_MM}, CUTTING_GAP_PG={CUTTING_GAP_PG}")
except ImportError:
    # 如果无法导入，使用默认值（向后兼容）
    PIXEL_RESOLUTION = 0.5  # 像素精度（mm/pixel）
    DEFAULT_WIDTH_MM = 100.0  # 默认基板宽度（mm）
    DEFAULT_HEIGHT_MM = 50.0  # 默认基板高度（mm）
    print("[WARNING] 无法从main.py导入全局配置，使用默认值")

# -------------------------- 工具函数（完整复制自trial3.py） --------------------------
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
        
        # 处理点可能是元组或有点对象的情况
        if hasattr(first, 'x') and hasattr(first, 'y'):
            # 点对象具有.x和.y属性
            return (first.x, first.y), (last.x, last.y)
        else:
            # 点可能是元组或列表
            try:
                # 尝试作为序列访问
                return (float(first[0]), float(first[1])), (float(last[0]), float(last[1]))
            except (IndexError, TypeError):
                # 如果无法访问，返回None
                return None
    
    elif entity_type == "POLYLINE":
        vertices = list(entity.vertices)
        if len(vertices) == 0:
            return None
        
        first = vertices[0]
        last = vertices[-1]
        
        # 处理点可能是有点对象或元组的情况
        def get_point_coords(point):
            """安全获取点的坐标"""
            if hasattr(point, 'dxf') and hasattr(point.dxf, 'location'):
                # 标准DXF点对象
                if hasattr(point.dxf.location, 'x') and hasattr(point.dxf.location, 'y'):
                    return (point.dxf.location.x, point.dxf.location.y)
            elif hasattr(point, 'x') and hasattr(point, 'y'):
                # 简单点对象
                return (point.x, point.y)
            else:
                # 尝试作为序列访问
                try:
                    return (float(point[0]), float(point[1]))
                except (IndexError, TypeError):
                    return None
        
        first_coords = get_point_coords(first)
        last_coords = get_point_coords(last)
        
        if first_coords is None or last_coords is None:
            return None
        
        # 对于封闭的POLYLINE，起点和终点应该是同一个点
        if entity.is_closed:
            return first_coords, first_coords
        else:
            return first_coords, last_coords
    
    elif entity_type == "SPLINE":
        # 对于样条线，需要检查是否为封闭样条线
        # 首先获取控制点或拟合点
        if entity.fit_points:
            points = entity.fit_points
        else:
            points = entity.control_points

        if len(points) == 0:
            return None

        # 检查样条线是否封闭
        is_closed = False
        if hasattr(entity, 'closed'):
            is_closed = entity.closed
        elif hasattr(entity, 'is_closed'):
            is_closed = entity.is_closed
        
        # 获取第一个和最后一个点
        first = points[0]
        last = points[-1]
        
        # 处理点可能是numpy数组的情况
        def get_point_coords(point):
            if hasattr(point, 'x') and hasattr(point, 'y'):
                # 点对象具有.x和.y属性
                return (point.x, point.y)
            else:
                # 点可能是numpy数组或元组
                try:
                    # 尝试作为序列访问
                    return (float(point[0]), float(point[1]))
                except (IndexError, TypeError):
                    return None
        
        first_coords = get_point_coords(first)
        last_coords = get_point_coords(last)
        
        if first_coords is None or last_coords is None:
            return None
        
        # 如果样条线是封闭的，起点和终点应该是同一个点
        if is_closed:
            return first_coords, first_coords
        else:
            return first_coords, last_coords
    
    elif entity_type == "ELLIPSE":
        # 对于椭圆，根据参数计算起点和终点
        center = entity.dxf.center
        major_axis = entity.dxf.major_axis
        ratio = entity.dxf.ratio
        start_param = entity.dxf.start_param
        end_param = entity.dxf.end_param
        
        # 计算短轴向量
        minor_axis = (-major_axis[1] * ratio, major_axis[0] * ratio)
        
        # 计算起点坐标
        start_x = center.x + major_axis[0] * math.cos(start_param) + minor_axis[0] * math.sin(start_param)
        start_y = center.y + major_axis[1] * math.cos(start_param) + minor_axis[1] * math.sin(start_param)
        
        # 计算终点坐标
        end_x = center.x + major_axis[0] * math.cos(end_param) + minor_axis[0] * math.sin(end_param)
        end_y = center.y + major_axis[1] * math.cos(end_param) + minor_axis[1] * math.sin(end_param)
        
        return (start_x, start_y), (end_x, end_y)
    
    elif entity_type == "CIRCLE":
        # 对于圆，起点和终点是同一个点（圆上的任意一点）
        center = entity.dxf.center
        radius = entity.dxf.radius
        
        # 选择圆上的一个点作为起点和终点（例如，0度方向）
        start_x = center.x + radius
        start_y = center.y
        
        return (start_x, start_y), (start_x, start_y)
    
    return None

def points_equal(p1, p2, tolerance=0.01):
    """Check if two points are equal (within tolerance range)"""
    # 增加容差值以处理浮点数精度问题
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5 < tolerance

def find_connected_chains(entities, close_tolerance=5.0):
    """查找首尾相连的实体链（改进版，支持双向查找）
    
    参数:
        entities: 实体列表
        close_tolerance: 链被认为"接近封闭"的最大距离（单位：与坐标相同）
                        默认5.0，对于大多数机械图纸足够
    """
    chains = []
    used = set()
    
    for i, entity in enumerate(entities):
        if i in used:
            continue
        
        endpoints = get_entity_endpoints(entity)
        if not endpoints:
            continue
            
        # 开始构建链 - 现在支持双向扩展
        chain = [entity]
        used.add(i)
        
        # 获取链的当前起点和终点
        chain_start, chain_end = endpoints
        
        # 标志是否进行了扩展
        expanded = True
        
        while expanded:
            expanded = False
            
            # 向前查找：从链的终点查找连接的实体
            for j, next_entity in enumerate(entities):
                if j in used:
                    continue
                    
                next_endpoints = get_entity_endpoints(next_entity)
                if not next_endpoints:
                    continue
                    
                next_start, next_end = next_endpoints
                
                # 检查是否连接到链的终点
                if points_equal(chain_end, next_start, 0.01):
                    chain.append(next_entity)
                    used.add(j)
                    chain_end = next_end
                    expanded = True
                    break
                elif points_equal(chain_end, next_end, 0.01):
                    # 需要反转实体方向
                    chain.append(next_entity)
                    used.add(j)
                    chain_end = next_start
                    expanded = True
                    break
            
            # 如果向前查找没有找到，尝试向后查找：从链的起点查找连接的实体
            if not expanded:
                for j, next_entity in enumerate(entities):
                    if j in used:
                        continue
                        
                    next_endpoints = get_entity_endpoints(next_entity)
                    if not next_endpoints:
                        continue
                        
                    next_start, next_end = next_endpoints
                    
                    # 检查是否连接到链的起点
                    if points_equal(chain_start, next_end, 0.01):
                        # 添加到链的开头（需要反转方向）
                        chain.insert(0, next_entity)
                        used.add(j)
                        chain_start = next_start
                        expanded = True
                        break
                    elif points_equal(chain_start, next_start, 0.01):
                        # 添加到链的开头
                        chain.insert(0, next_entity)
                        used.add(j)
                        chain_start = next_end
                        expanded = True
                        break
        
        # 检查链是否封闭（首尾相连）
        if points_equal(chain_start, chain_end):
            chains.append(chain)
            # print(f"  找到封闭链: {len(chain)} 个实体")
        else:
            # 检查是否接近封闭（对于浮点数精度问题或微小间隙）
            close_distance = ((chain_start[0] - chain_end[0])**2 + (chain_start[1] - chain_end[1])**2)**0.5
            if close_distance < close_tolerance and len(chain) > 1:
                # 链接近封闭，添加到结果中
                chains.append(chain)
                print(f"  找到接近封闭的链: {len(chain)} 个实体, 起点-终点距离={close_distance:.4f}")
            else:
                # 链不封闭且距离较大，可能是开口轮廓或错误
                # 可以将未使用的实体标记为未使用，以便其他链使用
                # 但为了简单起见，我们暂时只处理封闭或接近封闭的链
                if len(chain) > 1:
                    print(f"  链不封闭: {len(chain)} 个实体, 起点-终点距离={close_distance:.4f} (大于容差{close_tolerance})")
                pass
    
    return chains

def chain_to_polygon(chain):
    """将实体链转换为多边形"""
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
            if points_equal(previous_point, start, tolerance=0.01):
                # 正常方向：添加终点
                points.append(end)
                previous_point = end
            elif points_equal(previous_point, end, tolerance=0.01):
                # 反转方向：添加起点
                points.append(start)
                previous_point = start
            else:
                # 无法连接，添加终点作为后备
                points.append(end)
                previous_point = end
        
        elif entity_type == "ARC":
            # 离散化圆弧
            center = (entity.dxf.center.x, entity.dxf.center.y)
            radius = entity.dxf.radius
            start_angle = entity.dxf.start_angle
            end_angle = entity.dxf.end_angle
            
            # 确保角度方向正确
            if end_angle < start_angle:
                end_angle += 360
            
            # 采样圆弧点
            num_segments = max(8, int((end_angle - start_angle) / 5))  # 每5度一个点
            arc_points = []
            for k in range(num_segments + 1):
                angle = start_angle + (end_angle - start_angle) * k / num_segments
                x = center[0] + radius * math.cos(math.radians(angle))
                y = center[1] + radius * math.sin(math.radians(angle))
                arc_points.append((x, y))
            
            if not points:
                # 第一个点
                points.append(arc_points[0])
                previous_point = arc_points[0]
                # 添加剩余点
                for i in range(1, len(arc_points)):
                    points.append(arc_points[i])
                    previous_point = arc_points[i]
            else:
                # 检查连接点
                if points_equal(previous_point, arc_points[0], tolerance=0.01):
                    # 正常方向
                    for i in range(1, len(arc_points)):
                        points.append(arc_points[i])
                        previous_point = arc_points[i]
                elif points_equal(previous_point, arc_points[-1], tolerance=0.01):
                    # 反转方向
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
            else:
                # 检查连接点
                if points_equal(previous_point, vertex_points[0], tolerance=0.01):
                    # 正常方向
                    for i in range(1, len(vertex_points)):
                        points.append(vertex_points[i])
                        previous_point = vertex_points[i]
                elif points_equal(previous_point, vertex_points[-1], tolerance=0.01):
                    # 反转方向
                    for i in range(len(vertex_points)-2, -1, -1):
                        points.append(vertex_points[i])
                        previous_point = vertex_points[i]
                else:
                    # 无法连接，添加所有点
                    for point in vertex_points:
                        points.append(point)
                        previous_point = point
        
        elif entity_type == "ELLIPSE":
            # 离散化椭圆弧
            center = entity.dxf.center
            major_axis = entity.dxf.major_axis
            ratio = entity.dxf.ratio
            start_param = entity.dxf.start_param
            end_param = entity.dxf.end_param
            
            # 计算短轴向量
            minor_axis = (-major_axis[1] * ratio, major_axis[0] * ratio)
            
            # 采样椭圆弧点
            num_segments = max(8, int(abs(end_param - start_param) * 10))  # 每0.1弧度一个点
            ellipse_points = []
            for k in range(num_segments + 1):
                t = k / num_segments
                angle = start_param + t * (end_param - start_param)
                x = center.x + major_axis[0] * math.cos(angle) + minor_axis[0] * math.sin(angle)
                y = center.y + major_axis[1] * math.cos(angle) + minor_axis[1] * math.sin(angle)
                ellipse_points.append((x, y))
            
            if not points:
                # 第一个点
                points.append(ellipse_points[0])
                previous_point = ellipse_points[0]
                # 添加剩余点
                for i in range(1, len(ellipse_points)):
                    points.append(ellipse_points[i])
                    previous_point = ellipse_points[i]
            else:
                # 检查连接点
                if points_equal(previous_point, ellipse_points[0], tolerance=0.01):
                    # 正常方向
                    for i in range(1, len(ellipse_points)):
                        points.append(ellipse_points[i])
                        previous_point = ellipse_points[i]
                elif points_equal(previous_point, ellipse_points[-1], tolerance=0.01):
                    # 反转方向
                    for i in range(len(ellipse_points)-2, -1, -1):
                        points.append(ellipse_points[i])
                        previous_point = ellipse_points[i]
                else:
                    # 无法连接，添加所有点
                    for point in ellipse_points:
                        points.append(point)
                        previous_point = point
        
        elif entity_type == "CIRCLE":
            # 离散化圆
            center = entity.dxf.center
            radius = entity.dxf.radius
            
            # 采样圆点（36个点）
            num_points = 36
            circle_points = []
            for k in range(num_points):
                angle = 2 * math.pi * k / num_points
                x = center.x + radius * math.cos(angle)
                y = center.y + radius * math.sin(angle)
                circle_points.append((x, y))
            
            # 闭合圆
            circle_points.append(circle_points[0])
            
            if not points:
                # 第一个点
                points.append(circle_points[0])
                previous_point = circle_points[0]
                # 添加剩余点
                for i in range(1, len(circle_points)):
                    points.append(circle_points[i])
                    previous_point = circle_points[i]
            else:
                # 检查连接点
                if points_equal(previous_point, circle_points[0], tolerance=0.01):
                    # 正常方向
                    for i in range(1, len(circle_points)):
                        points.append(circle_points[i])
                        previous_point = circle_points[i]
                elif points_equal(previous_point, circle_points[-1], tolerance=0.01):
                    # 反转方向
                    for i in range(len(circle_points)-2, -1, -1):
                        points.append(circle_points[i])
                        previous_point = circle_points[i]
                else:
                    # 无法连接，添加所有点
                    for point in circle_points:
                        points.append(point)
                        previous_point = point
        
        elif entity_type == "SPLINE":
            # 离散化样条线
            if entity.fit_points:
                spline_points = entity.fit_points
            else:
                spline_points = entity.control_points
            
            # 处理点可能是numpy数组的情况
            vertex_points = []
            for point in spline_points:
                if hasattr(point, 'x') and hasattr(point, 'y'):
                    # 点对象具有.x和.y属性
                    vertex_points.append((point.x, point.y))
                else:
                    # 点可能是numpy数组或元组
                    try:
                        # 尝试作为数组访问
                        vertex_points.append((float(point[0]), float(point[1])))
                    except (IndexError, TypeError):
                        # 如果无法访问，跳过这个点
                        continue
            
            if not vertex_points:
                continue
                
            if not points:
                # 第一个顶点
                points.append(vertex_points[0])
                previous_point = vertex_points[0]
                # 添加剩余顶点
                for i in range(1, len(vertex_points)):
                    points.append(vertex_points[i])
                    previous_point = vertex_points[i]
            else:
                # 检查连接点
                if points_equal(previous_point, vertex_points[0], tolerance=0.01):
                    # 正常方向
                    for i in range(1, len(vertex_points)):
                        points.append(vertex_points[i])
                        previous_point = vertex_points[i]
                elif points_equal(previous_point, vertex_points[-1], tolerance=0.01):
                    # 反转方向
                    for i in range(len(vertex_points)-2, -1, -1):
                        points.append(vertex_points[i])
                        previous_point = vertex_points[i]
                else:
                    # 无法连接，添加所有点
                    for point in vertex_points:
                        points.append(point)
                        previous_point = point
    
    # 创建多边形
    try:
        if len(points) >= 3:
            poly = Polygon(points)
            return make_valid(poly) if not poly.is_valid else poly
    except:
        pass
    
    return None

def spline_to_polygon(spline, step=SAMPLING_STEP):
    """将样条线离散化为多边形（用于包含关系判断）"""
    # 检查样条线是否封闭 - 尝试多种属性
    is_closed = False
    
    # 方法1：检查 closed 属性（ezdxf常用）
    if hasattr(spline, 'closed') and spline.closed:
        is_closed = True
        print(f"  样条线通过closed属性识别为封闭：{spline.closed}")
    
    # 方法2：检查 is_closed 属性
    elif hasattr(spline, 'is_closed') and spline.is_closed:
        is_closed = True
        print(f"  样条线通过is_closed属性识别为封闭：{spline.is_closed}")
    
    # 方法3：检查DXF flags属性（位1表示封闭）
    elif hasattr(spline.dxf, 'flags'):
        flags = spline.dxf.flags
        if flags & 1:  # 位1表示封闭
            is_closed = True
            print(f"  样条线通过DXF flags识别为封闭：flags={flags} (二进制：{bin(flags)})")
    
    # 方法4：通过端点检查是否封闭（对于由多个样条线组成的封闭轮廓）
    # 注意：对于数学封闭的样条线，端点可能不重合，所以这是最后的手段
    if not is_closed:
        endpoints = get_entity_endpoints(spline)
        if endpoints:
            start, end = endpoints
            if points_equal(start, end, tolerance=0.1):
                is_closed = True
                print(f"  样条线通过端点重合识别为封闭：起点={start}, 终点={end}")
    
    if not is_closed:
        print(f"  样条线未被识别为封闭，跳过处理")
        return None
    
    print(f"  样条线识别为封闭，开始离散化...")
    
    # 获取样条线的拟合点/控制点
    raw_points = []
    if spline.fit_points and len(spline.fit_points) > 0:
        raw_points = spline.fit_points
        print(f"  使用拟合点：{len(raw_points)}个")
    elif spline.control_points and len(spline.control_points) > 0:
        raw_points = spline.control_points
        print(f"  使用控制点：{len(raw_points)}个")
    else:
        print(f"  警告：样条线没有拟合点或控制点")
        return None
    
    # 将原始点转换为(x, y)坐标元组
    points = []
    for p in raw_points:
        if hasattr(p, 'x') and hasattr(p, 'y'):
            points.append((p.x, p.y))
        elif hasattr(p, '__getitem__'):
            try:
                x = p[0]
                y = p[1]
                points.append((x, y))
            except (IndexError, TypeError):
                print(f"  警告：无法解析点：{p}")
        else:
            print(f"  警告：无法解析点类型：{type(p)}")
    
    if len(points) < 3:
        print(f"  错误：点数不足（{len(points)}个），无法创建多边形")
        return None
    
    print(f"  成功解析{len(points)}个点")
    
    # 转换为shapely多边形（确保有效）
    try:
        # 处理点对象，提取x,y坐标
        point_coords = []
        for p in points:
            if hasattr(p, 'x') and hasattr(p, 'y'):
                point_coords.append((p.x, p.y))
            else:
                # 点可能是numpy数组或元组
                try:
                    point_coords.append((float(p[0]), float(p[1])))
                except (IndexError, TypeError, ValueError):
                    continue
        
        if len(point_coords) < 3:
            return None
            
        poly = Polygon(point_coords)
        return make_valid(poly) if not poly.is_valid else poly
    except Exception as e:
        print(f"警告：无法将样条线转换为多边形：{e}")
        return None

def lwpolyline_to_polygon(lwpoly):
    """将轻量级多段线转换为shapely多边形"""
    if not lwpoly.closed:
        return None
    
    points = list(lwpoly.vertices())
    if len(points) < 3:
        return None
    
    try:
        poly = Polygon([(p.x, p.y) for p in points])
        return make_valid(poly) if not poly.is_valid else poly
    except:
        return None


def ellipse_to_polygon(ellipse, num_points=36):
    """
    将椭圆实体转换为shapely多边形
    
    参数:
        ellipse: ezdxf椭圆实体
        num_points: 离散化点数，默认36
    
    返回:
        Polygon: 转换后的多边形，如果转换失败则返回None
        注意：对于椭圆弧（部分椭圆），返回的多边形可能不封闭
    """
    try:
        center = ellipse.dxf.center
        major_axis = ellipse.dxf.major_axis
        ratio = ellipse.dxf.ratio
        start_param = ellipse.dxf.start_param
        end_param = ellipse.dxf.end_param
        param_range = end_param - start_param
        
        # 计算短轴向量：短轴垂直于长轴，长度为长轴长度 * ratio
        # 短轴向量 = (-major_axis.y * ratio, major_axis.x * ratio)
        minor_axis = (-major_axis[1] * ratio, major_axis[0] * ratio)
        
        # 将椭圆离散化为点集
        points = []
        
        # 对于完整椭圆，创建封闭多边形
        if abs(param_range - 2 * math.pi) < 0.001:
            # 完整椭圆
            for i in range(num_points):
                angle = 2 * math.pi * i / num_points
                # 椭圆参数方程
                x = center.x + major_axis[0] * math.cos(angle) + minor_axis[0] * math.sin(angle)
                y = center.y + major_axis[1] * math.cos(angle) + minor_axis[1] * math.sin(angle)
                points.append((x, y))
            
            # 闭合多边形
            if points:
                points.append(points[0])
        else:
            # 椭圆弧（部分椭圆）
            # 使用更多点来保证精度
            arc_num_points = max(num_points, int(abs(param_range) * 20))
            for i in range(arc_num_points + 1):
                t = i / arc_num_points
                angle = start_param + t * param_range
                # 椭圆参数方程
                x = center.x + major_axis[0] * math.cos(angle) + minor_axis[0] * math.sin(angle)
                y = center.y + major_axis[1] * math.cos(angle) + minor_axis[1] * math.sin(angle)
                points.append((x, y))
        
        if len(points) < 3:
            return None
        
        # 对于椭圆弧，我们仍然创建多边形（尽管可能不封闭）
        # 在实际使用中，椭圆弧应该通过find_connected_chains与其他实体连接
        poly = Polygon(points)
        return make_valid(poly) if not poly.is_valid else poly
        
    except Exception as e:
        print(f"警告：无法将椭圆转换为多边形：{e}")
        return None

def extract_closed_contours(msp):
    """从模型空间提取所有封闭轮廓（外轮廓/内孔）"""
    closed_contours = []
    
    # 首先，收集所有可能构成轮廓的实体
    contour_entities = []
    for entity in msp:
        entity_type = entity.dxftype()
        if entity_type in ["LINE", "ARC", "LWPOLYLINE", "POLYLINE", "SPLINE", "CIRCLE", "ELLIPSE"]:
            contour_entities.append(entity)
    
    print(f"找到 {len(contour_entities)} 个可能构成轮廓的实体")
    
    # 1. 首先处理原本就封闭的实体
    for entity in contour_entities:
        entity_type = entity.dxftype()
        poly = None
        
        # 处理轻量级多段线（最常见的封闭轮廓）
        if entity_type == "LWPOLYLINE":
            poly = lwpolyline_to_polygon(entity)
        
        # 处理样条线（复杂外轮廓）
        elif entity_type == "SPLINE":
            poly = spline_to_polygon(entity)
            if poly:
                print(f"  找到封闭样条线，面积: {poly.area:.2f}")
            else:
                # 尝试检查是否为封闭样条线
                if hasattr(entity, 'closed'):
                    print(f"  样条线 closed 属性: {entity.closed}")
                if hasattr(entity, 'is_closed'):
                    print(f"  样条线 is_closed 属性: {entity.is_closed}")
        
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
        
        # 处理椭圆（ELLIPSE实体）
        elif entity_type == "ELLIPSE":
            # 检查是否为完整椭圆（封闭椭圆）
            # 椭圆弧（部分椭圆）应该通过find_connected_chains与其他实体连接
            try:
                start_param = entity.dxf.start_param
                end_param = entity.dxf.end_param
                param_range = end_param - start_param
                is_full_ellipse = abs(param_range - 2 * math.pi) < 0.001
                
                if is_full_ellipse:
                    poly = ellipse_to_polygon(entity)
                    if poly:
                        print(f"  找到完整椭圆，面积: {poly.area:.2f}")
                else:
                    # 椭圆弧，不单独处理，留给find_connected_chains
                    poly = None
            except:
                # 如果无法获取参数，默认当作完整椭圆处理
                poly = ellipse_to_polygon(entity)
                if poly:
                    print(f"  找到椭圆（参数未知），面积: {poly.area:.2f}")
        
        # 过滤无效/微小轮廓
        if poly and poly.area > AREA_THRESHOLD:
            closed_contours.append({
                "entity": entity,
                "polygon": poly,
                "area": poly.area,
                "type": "single"
            })
    
    print(f"找到 {len(closed_contours)} 个原本封闭的轮廓")
    
    # 2. 查找由首尾相连的实体组成的封闭轮廓
    # 使用较大的容差值（30.0）以处理浮点数精度问题和微小间隙
    # 用户确认在CAD软件中所有轮廓都是封闭的
    chains = find_connected_chains(contour_entities, close_tolerance=30.0)
    print(f"找到 {len(chains)} 个首尾相连的实体链")
    
    for chain in chains:
        poly = chain_to_polygon(chain)
        if poly and poly.area > AREA_THRESHOLD:
            # 创建一个虚拟实体来表示整个链
            closed_contours.append({
                "entity": chain[0],  # 使用链中的第一个实体作为代表
                "polygon": poly,
                "area": poly.area,
                "type": "chain",
                "chain": chain  # 保存整个链供后续使用
            })
    
    print(f"总共找到 {len(closed_contours)} 个封闭轮廓")
    
    # 按面积降序排序（大轮廓优先作为外轮廓）
    closed_contours.sort(key=lambda x: x["area"], reverse=True)
    return closed_contours

def group_contours_by_part(contours):
    """根据包含关系分组轮廓（外轮廓+内孔=一个零件）"""
    parts = []
    used_contours = set()  # 标记已分配的轮廓
    
    # 首先过滤掉太小的轮廓（可能是内孔或无效轮廓）
    valid_contours = []
    for idx, contour in enumerate(contours):
        area = contour["area"]
        # 面积小于阈值的轮廓可能是内孔，但我们仍然保留它们用于包含关系判断
        valid_contours.append((idx, contour))
    
    # 按面积降序排序（大轮廓优先作为外轮廓）
    valid_contours.sort(key=lambda x: x[1]["area"], reverse=True)
    
    # 重新索引
    sorted_indices = [idx for idx, _ in valid_contours]
    sorted_contours = [contour for _, contour in valid_contours]
    
    for sorted_idx, (orig_idx, outer_candidate) in enumerate(valid_contours):
        if orig_idx in used_contours:
            continue
        
        outer_poly = outer_candidate["polygon"]
        outer_area = outer_candidate["area"]
        
        # 判断是否为外轮廓：不被任何更大的轮廓包含
        is_outer = True
        for j in range(sorted_idx):
            j_orig_idx, j_contour = valid_contours[j]
            if j_orig_idx in used_contours:
                continue
            
            larger_poly = j_contour["polygon"]
            larger_area = j_contour["area"]
            
            # 使用更宽松的包含关系判断
            # 1. 检查outer_poly是否完全在larger_poly内部
            # 2. 或者检查outer_poly的中心点是否在larger_poly内部
            if larger_poly.contains(outer_poly):
                is_outer = False
                break
            elif outer_poly.centroid.within(larger_poly):
                # 中心点在内部，但多边形可能不完全在内部（由于边界情况）
                is_outer = False
                break
        
        if not is_outer:
            continue
        
        # 收集该外轮廓包含的所有内孔
        holes = []
        for j, (hole_orig_idx, hole_candidate) in enumerate(valid_contours):
            if hole_orig_idx == orig_idx or hole_orig_idx in used_contours:
                continue
            
            hole_poly = hole_candidate["polygon"]
            hole_area = hole_candidate["area"]
            
            # 内孔应该比外轮廓小
            if hole_area >= outer_area * 0.8:  # 面积接近的轮廓不太可能是内孔
                continue
            
            # 使用多种方法检查包含关系
            is_hole = False
            
            # 方法1：严格包含
            if outer_poly.contains(hole_poly):
                is_hole = True
            # 方法2：中心点在内部
            elif hole_poly.centroid.within(outer_poly):
                is_hole = True
            # 方法3：使用buffer处理边界情况
            elif outer_poly.buffer(0.1).contains(hole_poly.buffer(-0.1)):
                is_hole = True
            
            if is_hole:
                holes.append(hole_candidate)
                used_contours.add(hole_orig_idx)
        
        # 计算最小外包矩形中心点（而不是形心）
        min_x, min_y, max_x, max_y = outer_poly.bounds
        bbox_center_x = (min_x + max_x) / 2.0
        bbox_center_y = (min_y + max_y) / 2.0
        
        # 封装零件数据
        parts.append({
            "outer": outer_candidate,
            "holes": holes,
            "center": (bbox_center_x, bbox_center_y),  # 最小外包矩形中心点
            "centroid": (outer_poly.centroid.x, outer_poly.centroid.y),  # 保留形心供参考
            "bbox": (min_x, min_y, max_x, max_y)  # 边界框信息
        })
        used_contours.add(orig_idx)
    
    return parts

def create_part_blocks(doc, parts):
    """为每个零件创建独立的Block，并插入到模型空间"""
    msp = doc.modelspace()
    
    for part_idx, part in enumerate(parts):
        # 1. 创建Block
        block_name = f"{PART_BLOCK_PREFIX}{part_idx:03d}"
        if block_name in doc.blocks:
            doc.blocks.delete(block_name)  # 覆盖已存在的块
        block = doc.blocks.new(name=block_name)
        
        # 2. 添加外轮廓到Block
        outer_contour = part["outer"]
        outer_type = outer_contour.get("type", "single")
        
        if outer_type == "chain":
            # 对于链类型的轮廓，添加整个链中的所有实体
            chain = outer_contour.get("chain", [])
            for entity in chain:
                try:
                    block.add_entity(entity.copy())
                except Exception as e:
                    print(f"警告：无法复制链中的实体 {entity.dxftype()}：{e}")
        else:
            # 对于单个实体类型的轮廓
            outer_entity = outer_contour["entity"]
            try:
                block.add_entity(outer_entity.copy())
            except Exception as e:
                print(f"警告：无法复制外轮廓实体：{e}")
        
        # 3. 添加内孔到Block
        hole_count = 0
        for hole in part["holes"]:
            hole_type = hole.get("type", "single")
            
            if hole_type == "chain":
                # 对于链类型的孔，添加整个链中的所有实体
                chain = hole.get("chain", [])
                for entity in chain:
                    try:
                        block.add_entity(entity.copy())
                        hole_count += 1
                    except Exception as e:
                        print(f"警告：无法复制孔链中的实体 {entity.dxftype()}：{e}")
            else:
                # 对于单个实体类型的孔
                hole_entity = hole["entity"]
                try:
                    block.add_entity(hole_entity.copy())
                    hole_count += 1
                except Exception as e:
                    print(f"警告：无法复制孔实体：{e}")
        
        # 4. 在模型空间插入Block引用（基于零件中心点）
        try:
            msp.add_blockref(
                block_name,
                insert=part["center"],  # 插入点为零件中心点
                dxfattribs={"layer": "PARTS_BLOCKS"}  # 块引用单独图层
            )
        except Exception as e:
            print(f"警告：无法插入块引用 {block_name}：{e}")
            continue
        
        # 统计实际添加的实体数量
        outer_entity_count = len(outer_contour.get("chain", [])) if outer_type == "chain" else 1
        print(f"已创建零件块 {block_name}：外轮廓实体{outer_entity_count}个，内孔{hole_count}个")

# -------------------------- 可视化函数（完整复制自trial5.py） --------------------------
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
    
    改进：对于样条线轮廓，使用更精确的绘制方法
    """
    # 检查外轮廓是否为单个样条线实体
    outer_contour = part["outer"]
    outer_entity = outer_contour.get("entity")
    outer_type = outer_contour.get("type", "single")
    
    # 如果是单个样条线实体，使用精确绘制方法
    if outer_type == "single" and outer_entity and outer_entity.dxftype() == "SPLINE":
        # 使用draw_entity_ordered绘制样条线
        draw_entity_ordered(ax, outer_entity, 0, color='black', linewidth=2, alpha=1.0)
        print(f"  零件#{part_index}: 使用精确样条线绘制")
    else:
        # 使用原始的多边形绘制方法
        outer_poly = outer_contour["polygon"]
        
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
        hole_entity = hole.get("entity")
        hole_type = hole.get("type", "single")
        
        # 如果是单个样条线孔，使用精确绘制方法
        if hole_type == "single" and hole_entity and hole_entity.dxftype() == "SPLINE":
            draw_entity_ordered(ax, hole_entity, hole_idx, color='red', linewidth=1.5, alpha=1.0)
        else:
            # 使用原始的多边形绘制方法
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
    ax.plot(center[0], center[1], 'ro', markersize=6)
    
    # Add index text (simplified)
    # 计算边界框
    outer_poly = part["outer"]["polygon"]
    if hasattr(outer_poly, 'bounds'):
        bbox = outer_poly.bounds
    else:
        # 如果没有bounds属性，使用外轮廓的坐标计算
        if hasattr(outer_poly, 'exterior'):
            coords = list(outer_poly.exterior.coords)
        else:
            coords = []
        if coords:
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            bbox = (min(xs), min(ys), max(xs), max(ys))
        else:
            bbox = (center[0]-10, center[1]-10, center[0]+10, center[1]+10)
    
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
        entity: DXF实体 (LINE, ARC, POLYLINE, LWPOLYLINE, SPLINE, ELLIPSE)
        entity_index: 实体索引（用于线条编号和线型选择）
    
    返回:
        list: 有序点序列 [(x1, y1), (x2, y2), ...]
        str: 实体类型
    
    处理逻辑:
        1. LINE: 直接使用起点和终点
        2. ARC: 离散化为点序列，保持角度顺序
        3. POLYLINE/LWPOLYLINE: 提取所有顶点，保持原始顺序
        4. SPLINE: 使用spline_to_points函数离散化，保持样条线参数顺序
        5. ELLIPSE: 离散化为点序列，保持参数顺序
        6. 使用最近邻算法确保点按轮廓顺序连接
    
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
    
    elif entity_type == "ELLIPSE":
        # 椭圆：离散化为点序列，保持参数顺序
        center = entity.dxf.center
        major_axis = entity.dxf.major_axis
        ratio = entity.dxf.ratio
        start_param = entity.dxf.start_param
        end_param = entity.dxf.end_param
        param_range = end_param - start_param
        
        # 计算短轴向量：短轴垂直于长轴，长度为长轴长度 * ratio
        # 短轴向量 = (-major_axis.y * ratio, major_axis.x * ratio)
        minor_axis = (-major_axis[1] * ratio, major_axis[0] * ratio)
        
        # 根据参数范围决定点数
        # 对于完整椭圆，使用36个点；对于椭圆弧，根据弧长决定点数
        if abs(param_range - 2 * math.pi) < 0.001:
            # 完整椭圆
            num_points = 36
            for i in range(num_points):
                angle = 2 * math.pi * i / num_points
                # 椭圆参数方程
                x = center.x + major_axis[0] * math.cos(angle) + minor_axis[0] * math.sin(angle)
                y = center.y + major_axis[1] * math.cos(angle) + minor_axis[1] * math.sin(angle)
                points.append((x, y))
            # 闭合椭圆
            if points:
                points.append(points[0])
        else:
            # 椭圆弧（部分椭圆）
            # 根据弧长决定点数（近似计算）
            # 椭圆弧长近似：使用平均半径
            avg_radius = math.sqrt(major_axis[0]**2 + major_axis[1]**2) * (1 + ratio) / 2
            arc_length = avg_radius * abs(param_range)
            num_points = max(5, int(arc_length / 2))  # 每2mm一个点
            num_points = min(num_points, 50)  # 限制最大点数
            
            for i in range(num_points + 1):
                t = i / num_points
                angle = start_param + t * param_range
                # 椭圆参数方程
                x = center.x + major_axis[0] * math.cos(angle) + minor_axis[0] * math.sin(angle)
                y = center.y + major_axis[1] * math.cos(angle) + minor_axis[1] * math.sin(angle)
                points.append((x, y))
    
    # 对于ARC、SPLINE和ELLIPSE，点已经按轮廓顺序排列，不需要重新排序
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
    ax.plot(center[0], center[1], 'ro', markersize=8, label='Center')
    
    # 添加索引文本
    # 计算边界框
    outer_poly = part["outer"]["polygon"]
    if hasattr(outer_poly, 'bounds'):
        bbox = outer_poly.bounds
    else:
        # 如果没有bounds属性，使用外轮廓的坐标计算
        if hasattr(outer_poly, 'exterior'):
            coords = list(outer_poly.exterior.coords)
        else:
            coords = []
        if coords:
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            bbox = (min(xs), min(ys), max(xs), max(ys))
        else:
            bbox = (center[0]-10, center[1]-10, center[0]+10, center[1]+10)
    
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
        print(f"  Part #{i}: Outer area={part['outer']['area']:.2f}, Holes count={len(part['holes'])}, Center=({part['center'][0]:.2f}, {part['center'][1]:.2f})")
    
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
        print(f"  Part #{i}: Outer area={part['outer']['area']:.2f}, Holes count={len(part['holes'])}, Center=({part['center'][0]:.2f}, {part['center'][1]:.2f})")
    
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

# -------------------------- 位图处理函数 --------------------------
def calculate_bounding_box(part):
    """
    计算零件的最小包围矩形（仅基于外轮廓）
    
    参数:
        part: 零件数据字典，包含outer和holes
    
    返回:
        tuple: (min_x, min_y, max_x, max_y, width, height)
    """
    # 获取外轮廓的多边形
    outer_poly = part["outer"]["polygon"]
    
    # 获取外轮廓的边界框（不包含内孔）
    min_x, min_y, max_x, max_y = outer_poly.bounds
    
    width = max_x - min_x
    height = max_y - min_y
    
    return min_x, min_y, max_x, max_y, width, height


def render_part_to_bitmap(part, pixel_resolution=PIXEL_RESOLUTION, margin_pixels=None, pg=CUTTING_GAP_PG):
    """
    将零件渲染为位图（二维矩阵） - 仅外轮廓，不包含内孔
    
    参数:
        part: 零件数据字典
        pixel_resolution: 像素精度（mm/pixel），默认0.1mm/pixel
        margin_pixels: 边缘留白像素数（如果为None，则根据pg自动计算）
        pg: 切割间隙（mm），默认4.0mm，用于计算动态边框宽度
    
    返回:
        tuple: (bitmap_matrix, width_pixels, height_pixels, bounds)
    """
    # 1. 计算包围矩形（仅基于外轮廓）
    outer_poly = part["outer"]["polygon"]
    min_x, min_y, max_x, max_y = outer_poly.bounds
    width_mm = max_x - min_x
    height_mm = max_y - min_y
    
    # 2. 计算边缘留白像素数
    if margin_pixels is None:
        # 根据用户需求：边框宽度 = 1 + pg/2 像素
        # pg/2 是偏移半径（mm），需要转换为像素
        offset_radius_mm = pg / 2.0
        offset_radius_pixels = int(offset_radius_mm / pixel_resolution)
        margin_pixels = 1 + offset_radius_pixels
        print(f"  自动计算边框宽度：1 + pg/2 = 1 + {offset_radius_pixels} = {margin_pixels} 像素")
    else:
        print(f"  使用指定边框宽度：{margin_pixels} 像素")
    
    # 3. 计算位图尺寸（添加边缘留白）
    width_pixels = int(width_mm / pixel_resolution) + 2 * margin_pixels
    height_pixels = int(height_mm / pixel_resolution) + 2 * margin_pixels
    
    # 3. 创建位图矩阵（初始为黑色0）
    bitmap = np.zeros((height_pixels, width_pixels), dtype=np.uint8)
    
    # 4. 坐标转换函数：从世界坐标到位图坐标
    def world_to_pixel(x, y):
        # 将世界坐标转换为相对于包围矩形左下角的坐标
        rel_x = x - min_x
        rel_y = y - min_y
        
        # 转换为像素坐标（添加边缘留白）
        pixel_x = int(rel_x / pixel_resolution) + margin_pixels
        pixel_y_raw = int(rel_y / pixel_resolution) + margin_pixels
        
        # 翻转y坐标：世界坐标系y向上为正，位图坐标系y向下为正
        pixel_y = height_pixels - 1 - pixel_y_raw
        
        # 确保在边界内
        pixel_x = max(0, min(pixel_x, width_pixels - 1))
        pixel_y = max(0, min(pixel_y, height_pixels - 1))
        
        return pixel_x, pixel_y
    
    # 5. 获取外轮廓坐标
    # 首先尝试从原始实体获取更高质量的点集（特别是对于样条线）
    exterior_coords = []
    outer_entity = part["outer"].get("entity")
    
    # 调试信息：检查实体类型
    if outer_entity:
        print(f"  外轮廓实体类型：{outer_entity.dxftype()}")
    else:
        print("  警告：外轮廓没有实体信息，将使用多边形坐标")
    
    if outer_entity and outer_entity.dxftype() == "SPLINE":
        print("  检测到样条线，使用精确样条线离散化方法...")
        # 对于样条线，使用与draw_part相同的精确离散化方法
        try:
            # 使用spline_to_points函数进行高质量离散化
            # 增加采样点数以获得更平滑的曲线
            # 根据样条线长度动态计算采样点数，最小50，最大200
            # 首先估算样条线长度
            try:
                # 获取样条线的控制点或拟合点来估算长度
                if outer_entity.fit_points and len(outer_entity.fit_points) > 0:
                    # 使用拟合点估算长度
                    points = outer_entity.fit_points
                elif outer_entity.control_points and len(outer_entity.control_points) > 0:
                    # 使用控制点估算长度
                    points = outer_entity.control_points
                else:
                    points = []
                
                if len(points) >= 2:
                    # 简单估算：计算所有点之间的总距离
                    total_length = 0
                    for i in range(len(points)-1):
                        p1 = points[i]
                        p2 = points[i+1]
                        if hasattr(p1, 'x') and hasattr(p1, 'y'):
                            x1, y1 = p1.x, p1.y
                        else:
                            x1, y1 = float(p1[0]), float(p1[1])
                        if hasattr(p2, 'x') and hasattr(p2, 'y'):
                            x2, y2 = p2.x, p2.y
                        else:
                            x2, y2 = float(p2[0]), float(p2[1])
                        total_length += ((x2-x1)**2 + (y2-y1)**2)**0.5
                    
                    # 根据长度计算采样点数：每毫米2个点，最小50，最大200
                    num_points = int(total_length * 2)
                    num_points = max(50, min(num_points, 200))
                    print(f"  根据样条线估算长度 {total_length:.2f}mm 计算采样点数：{num_points}")
                else:
                    # 无法估算长度，使用默认值
                    num_points = 100
                    print(f"  无法估算样条线长度，使用默认采样点数：{num_points}")
            except Exception as e:
                # 如果计算失败，使用合理的默认值
                num_points = 100
                print(f"  样条线长度估算失败，使用默认采样点数：{num_points}（错误：{e}）")
            
            # 检查样条线是否封闭
            closed = False
            if hasattr(outer_entity, 'closed') and outer_entity.closed:
                closed = True
                print(f"  样条线通过closed属性识别为封闭：{outer_entity.closed}")
            elif hasattr(outer_entity, 'is_closed') and outer_entity.is_closed:
                closed = True
                print(f"  样条线通过is_closed属性识别为封闭：{outer_entity.is_closed}")
            elif hasattr(outer_entity.dxf, 'flags'):
                flags = outer_entity.dxf.flags
                if flags & 1:  # 位1表示封闭
                    closed = True
                    print(f"  样条线通过DXF flags识别为封闭：flags={flags}")
            
            print(f"  样条线封闭状态：{closed}")
            
            # 使用spline_to_points进行离散化
            points = spline_to_points(outer_entity, num_points=num_points)
            
            if not points or len(points) < 3:
                print(f"  警告：spline_to_points返回点数不足（{len(points) if points else 0}个）")
                raise ValueError("样条线离散化失败")
            
            print(f"  样条线精确离散化成功：生成 {len(points)} 个点")
            
            # 对于封闭样条线，确保首尾点相同（闭合）
            if closed and len(points) >= 3:
                # 检查首尾点是否相同
                first = points[0]
                last = points[-1]
                import math
                distance = math.sqrt((last[0] - first[0])**2 + (last[1] - first[1])**2)
                if distance > 0.1:  # 如果首尾点不接近，添加第一个点作为最后一个点
                    points.append(first)
                    print(f"  封闭样条线：添加首点作为尾点以闭合曲线（距离={distance:.6f}）")
            
            # 使用离散化的点
            exterior_coords = points
            print(f"  使用 {len(exterior_coords)} 个精确样条线点进行位图渲染")
                
        except Exception as e:
            print(f"  样条线精确离散化失败：{e}")
            import traceback
            traceback.print_exc()
            # 回退到使用多边形坐标
            if hasattr(outer_poly, 'exterior'):
                exterior_coords = list(outer_poly.exterior.coords)
                print(f"  回退到多边形坐标：{len(exterior_coords)} 个点")
            else:
                # 处理多部分几何体
                exterior_coords = []
                if hasattr(outer_poly, 'boundary'):
                    boundary = outer_poly.boundary
                    if hasattr(boundary, 'geoms'):  # 多部分几何体（MultiPolygon等）
                        for geom in boundary.geoms:
                            if hasattr(geom, 'coords'):
                                exterior_coords.extend(list(geom.coords))
                    elif hasattr(boundary, 'coords'):  # 单部分几何体
                        exterior_coords = list(boundary.coords)
                print(f"  回退到边界坐标：{len(exterior_coords)} 个点")
    else:
        # 对于非样条线实体，使用多边形坐标
        if hasattr(outer_poly, 'exterior'):
            exterior_coords = list(outer_poly.exterior.coords)
            print(f"  使用多边形坐标（非样条线）：{len(exterior_coords)} 个点")
        else:
            # 处理多部分几何体
            exterior_coords = []
            if hasattr(outer_poly, 'boundary'):
                boundary = outer_poly.boundary
                if hasattr(boundary, 'geoms'):  # 多部分几何体（MultiPolygon等）
                    for geom in boundary.geoms:
                        if hasattr(geom, 'coords'):
                            exterior_coords.extend(list(geom.coords))
                elif hasattr(boundary, 'coords'):  # 单部分几何体
                    exterior_coords = list(boundary.coords)
            print(f"  使用边界坐标（非样条线）：{len(exterior_coords)} 个点")
    
    if not exterior_coords:
        print("  警告：没有找到任何坐标点")
        return bitmap, width_pixels, height_pixels, (min_x, min_y, max_x, max_y)
    
    print(f"  最终使用 {len(exterior_coords)} 个坐标点进行位图渲染")
    
    # 6. 使用matplotlib.path.Path进行精确填充
    from matplotlib.path import Path
    
    # 将坐标转换为像素坐标
    pixel_coords = [world_to_pixel(x, y) for x, y in exterior_coords]
    
    # 创建路径对象
    path = Path(pixel_coords, closed=True)
    
    # 创建网格点坐标
    x_coords = np.arange(width_pixels)
    y_coords = np.arange(height_pixels)
    X, Y = np.meshgrid(x_coords, y_coords)
    points = np.column_stack([X.ravel(), Y.ravel()])
    
    # 判断哪些点在多边形内
    mask = path.contains_points(points)
    mask = mask.reshape((height_pixels, width_pixels))
    
    # 将多边形内的点设置为白色
    bitmap[mask] = 255
    
    # 7. 可选：绘制多边形边界（轮廓线）
    # 这可以提供更好的视觉效果，但会增加计算量
    # 如果需要轮廓线，可以取消注释以下代码
    """
    # 绘制多边形边界
    for i in range(len(pixel_coords)):
        x1, y1 = pixel_coords[i]
        x2, y2 = pixel_coords[(i + 1) % len(pixel_coords)]
        
        # 使用Bresenham算法绘制直线
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        while True:
            if 0 <= x1 < width_pixels and 0 <= y1 < height_pixels:
                bitmap[y1, x1] = 255
            
            if x1 == x2 and y1 == y2:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
    """
    
    return bitmap, width_pixels, height_pixels, (min_x, min_y, max_x, max_y)


def save_bitmap_as_image(bitmap_matrix, output_path):
    """
    将位图矩阵保存为图像文件
    
    参数:
        bitmap_matrix: 二维numpy数组
        output_path: 输出文件路径
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 使用matplotlib保存图像
        plt.figure(figsize=(10, 10))
        plt.imshow(bitmap_matrix, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        
        print(f"位图已保存：{output_path}")
        return True
    except Exception as e:
        print(f"保存位图时出错：{e}")
        return False


def save_bitmap_as_numpy(bitmap_matrix, output_path):
    """
    将位图矩阵保存为numpy文件
    
    参数:
        bitmap_matrix: 二维numpy数组
        output_path: 输出文件路径
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存为numpy文件
        np.save(output_path, bitmap_matrix)
        
        print(f"位图矩阵已保存：{output_path}.npy")
        return True
    except Exception as e:
        print(f"保存位图矩阵时出错：{e}")
        return False


def apply_offset_to_bitmap(bitmap, pixel_resolution=PIXEL_RESOLUTION, pg=CUTTING_GAP_PG, ensure_border_distance=False):
    """
    对位图轮廓进行向外偏移操作
    
    参数:
        bitmap: 原始位图矩阵（0=黑色背景，255=白色轮廓）
        pixel_resolution: 像素精度（mm/pixel），默认0.1mm/pixel
        pg: 切割间隙（mm），默认4mm
        ensure_border_distance: 是否保证轮廓到图像边界有至少1像素距离（默认False，根据用户需求）
    
    返回:
        numpy.ndarray: 偏移后的位图矩阵
    """
    print(f"  应用轮廓偏移：pg={pg}mm，像素精度={pixel_resolution}mm/pixel")
    
    # 计算偏移半径（像素）
    offset_radius_mm = pg / 2.0  # 偏移距离为pg/2
    offset_radius_pixels = int(offset_radius_mm / pixel_resolution)
    print(f"  偏移距离：{offset_radius_mm}mm，对应{offset_radius_pixels}像素")
    
    if offset_radius_pixels <= 0:
        print(f"  警告：偏移半径太小（{offset_radius_pixels}像素），跳过偏移操作")
        return bitmap
    
    # 获取位图尺寸
    height, width = bitmap.shape
    
    # 检查是否有白色像素
    if np.sum(bitmap == 255) == 0:
        print(f"  警告：位图中没有白色像素，跳过偏移操作")
        return bitmap
    
    print(f"  使用高效欧式距离变换进行轮廓扩展...")
    
    # 尝试使用scipy的distance_transform_edt进行高效欧式距离计算
    scipy_available = False
    scipy_error_msg = ""
    
    try:
        # 先检查scipy是否可用
        import scipy
        scipy_version = scipy.__version__
        print(f"  检测到SciPy版本: {scipy_version}")
        
        # 检查NumPy版本兼容性
        numpy_version = np.__version__
        print(f"  检测到NumPy版本: {numpy_version}")
        
        # 检查兼容性
        # SciPy 1.14.0+ 支持 NumPy 2.x
        # 解析版本号
        def parse_version(version_str):
            parts = version_str.split('.')
            major = int(parts[0]) if parts[0].isdigit() else 0
            minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            patch = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
            return major, minor, patch
        
        scipy_major, scipy_minor, scipy_patch = parse_version(scipy_version)
        numpy_major, numpy_minor, numpy_patch = parse_version(numpy_version)
        
        # 判断是否兼容
        is_compatible = True
        
        if numpy_major >= 2:
            # NumPy 2.x 需要 SciPy 1.14.0+
            if scipy_major == 1 and scipy_minor < 14:
                is_compatible = False
                print(f"  警告：NumPy {numpy_version} 需要 SciPy 1.14.0+，当前为 {scipy_version}")
                print(f"  将使用简单膨胀方法以避免兼容性问题")
        
        if is_compatible:
            scipy_available = True
            print(f"  SciPy与NumPy版本兼容，将尝试使用高效距离变换")
        else:
            scipy_available = False
            scipy_error_msg = f"NumPy {numpy_version} 与 SciPy {scipy_version} 兼容性问题"
            
    except (ImportError, ModuleNotFoundError, AttributeError, RuntimeError) as e:
        print(f"  警告：scipy库不可用（错误：{e}），将使用简单膨胀方法")
        scipy_error_msg = str(e)
    
    if scipy_available:
        try:
            from scipy.ndimage import distance_transform_edt
            
            # 创建二值掩码：背景为True（距离计算需要），轮廓为False
            # 注意：distance_transform_edt计算到最近False像素的距离
            mask = bitmap == 0  # 背景为True
            
            # 计算欧式距离
            distances = distance_transform_edt(mask)
            
            # 创建偏移后的位图：距离小于等于偏移半径的位置设置为白色
            offset_bitmap = np.zeros_like(bitmap)
            offset_bitmap[distances <= offset_radius_pixels] = 255
            
            print(f"  高效距离变换完成：原始白色像素 {np.sum(bitmap == 255)} 个，偏移后白色像素 {np.sum(offset_bitmap == 255)} 个")
            
        except ImportError as e:
            print(f"  警告：无法导入distance_transform_edt（错误：{e}），使用简单膨胀方法")
            scipy_available = False
            scipy_error_msg = str(e)
        except RuntimeError as e:
            # 捕获NumPy兼容性相关的运行时错误
            if "numpy.core.multiarray" in str(e) or "NumPy" in str(e):
                print(f"  警告：NumPy兼容性问题（错误：{e}），使用简单膨胀方法")
                print(f"  解决方案：请降级NumPy到1.x版本或升级SciPy到支持NumPy 2.x的版本")
            else:
                print(f"  警告：scipy运行时错误（错误：{e}），使用简单膨胀方法")
            scipy_available = False
            scipy_error_msg = str(e)
        except Exception as e:
            print(f"  警告：scipy距离变换失败（错误：{e}），使用简单膨胀方法")
            scipy_available = False
            scipy_error_msg = str(e)
    
    if not scipy_available:
        # 使用简单的形态学膨胀作为备选方案
        # 这种方法不如欧式距离精确，但更简单快速
        # 为了允许膨胀超出边界，我们首先创建一个更大的位图
        expand_size = offset_radius_pixels  # 扩展大小等于偏移半径
        expanded_height = height + expand_size * 2
        expanded_width = width + expand_size * 2
        expanded_bitmap = np.zeros((expanded_height, expanded_width), dtype=bitmap.dtype)
        
        # 将原始位图放在扩展位图的中心
        expanded_bitmap[expand_size:expand_size+height, expand_size:expand_size+width] = bitmap
        
        # 对扩展位图进行膨胀
        if offset_radius_pixels > 0:
            print(f"  执行膨胀操作：半径={offset_radius_pixels}像素")
            
            # 方法1：使用卷积进行更精确的膨胀（如果可用）
            try:
                from scipy import ndimage
                
                # 创建结构元素（圆形近似）
                # 对于小半径，使用曼哈顿距离近似
                structure_size = offset_radius_pixels * 2 + 1
                structure = np.zeros((structure_size, structure_size), dtype=bool)
                
                # 创建圆形结构元素
                center = offset_radius_pixels
                for i in range(structure_size):
                    for j in range(structure_size):
                        distance = np.sqrt((i - center)**2 + (j - center)**2)
                        if distance <= offset_radius_pixels:
                            structure[i, j] = True
                
                # 使用二进制膨胀
                expanded_bitmap = ndimage.binary_dilation(
                    expanded_bitmap == 255,
                    structure=structure
                ).astype(expanded_bitmap.dtype) * 255
                
                print(f"  使用SciPy二进制膨胀完成")
                
            except (ImportError, AttributeError, RuntimeError):
                # 方法2：使用改进的膨胀算法（距离近似）
                print(f"  SciPy不可用，使用改进的膨胀算法")
                
                # 创建距离掩码
                # 对于每个像素，计算到最近白色像素的距离
                # 使用BFS（广度优先搜索）近似距离
                
                # 首先找到所有白色像素的位置
                white_pixels = np.argwhere(expanded_bitmap == 255)
                
                if len(white_pixels) > 0:
                    # 创建距离数组
                    distances = np.full((expanded_height, expanded_width), offset_radius_pixels + 1, dtype=np.int32)
                    
                    # 初始化队列
                    from collections import deque
                    queue = deque()
                    
                    # 将所有白色像素的距离设为0，并加入队列
                    for y, x in white_pixels:
                        distances[y, x] = 0
                        queue.append((y, x))
                    
                    # BFS计算距离
                    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                                 (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8方向
                    
                    while queue:
                        y, x = queue.popleft()
                        current_dist = distances[y, x]
                        
                        if current_dist >= offset_radius_pixels:
                            continue
                            
                        for dy, dx in directions:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < expanded_height and 0 <= nx < expanded_width:
                                new_dist = current_dist + 1
                                if new_dist < distances[ny, nx]:
                                    distances[ny, nx] = new_dist
                                    queue.append((ny, nx))
                    
                    # 根据距离创建膨胀后的位图
                    expanded_bitmap = np.zeros_like(expanded_bitmap)
                    expanded_bitmap[distances <= offset_radius_pixels] = 255
                    
                    print(f"  使用BFS距离近似膨胀完成")
                else:
                    print(f"  警告：没有找到白色像素，跳过膨胀")
        
        # 从扩展位图中提取膨胀后的轮廓区域
        # 原始位图在扩展位图中的位置是 [expand_size:expand_size+height, expand_size:expand_size+width]
        # 膨胀后，我们需要提取以原始位置为中心的区域，但包含膨胀扩展
        
        # 计算提取区域的边界
        # 起始位置：原始位置减去膨胀半径
        extract_y_start = expand_size - offset_radius_pixels
        extract_y_end = expand_size + height + offset_radius_pixels
        extract_x_start = expand_size - offset_radius_pixels
        extract_x_end = expand_size + width + offset_radius_pixels
        
        # 确保边界在扩展位图范围内
        extract_y_start = max(0, extract_y_start)
        extract_y_end = min(expanded_height, extract_y_end)
        extract_x_start = max(0, extract_x_start)
        extract_x_end = min(expanded_width, extract_x_end)
        
        # 提取膨胀后的轮廓区域
        offset_bitmap = expanded_bitmap[extract_y_start:extract_y_end, extract_x_start:extract_x_end]
        
        print(f"  简单膨胀完成：提取区域尺寸 {offset_bitmap.shape}，白色像素 {np.sum(offset_bitmap == 255)} 个")
        print(f"  注意：简单膨胀方法不如欧式距离精确，但作为备选方案可用")
        
        # 更新尺寸变量，以便后续的边界距离调整使用新尺寸
        height, width = offset_bitmap.shape
    
    # 如果需要保证边界距离
    if ensure_border_distance:
        print(f"  检查边界距离...")
        # 找到偏移后位图中的白色像素
        offset_white_pixels = np.argwhere(offset_bitmap == 255)
        
        if len(offset_white_pixels) > 0:
            # 检查是否太靠近边界
            min_x = np.min(offset_white_pixels[:, 1])
            max_x = np.max(offset_white_pixels[:, 1])
            min_y = np.min(offset_white_pixels[:, 0])
            max_y = np.max(offset_white_pixels[:, 0])
            
            print(f"  偏移后轮廓边界：x=[{min_x}, {max_x}], y=[{min_y}, {max_y}]，图像尺寸：{width}×{height}")
            
            # 检查是否需要调整 - 确保轮廓到各边都有至少一个像素距离
            # 轮廓应该在位置 [1, width-2] 和 [1, height-2] 范围内
            # 但我们需要考虑轮廓可能已经离边界足够远的情况
            target_min_x = 1  # 轮廓最左像素应该在x=1位置
            target_max_x = width - 2  # 轮廓最右像素应该在x=width-2位置
            target_min_y = 1  # 轮廓最上像素应该在y=1位置
            target_max_y = height - 2  # 轮廓最下像素应该在y=height-2位置
            
            # 检查当前边界是否在目标范围内
            # 如果轮廓已经在目标范围内，不需要调整
            needs_adjustment = False
            
            if min_x < target_min_x or max_x > target_max_x or min_y < target_min_y or max_y > target_max_y:
                needs_adjustment = True
                print(f"    需要调整：轮廓太靠近边界")
                print(f"              当前边界x=[{min_x}, {max_x}], y=[{min_y}, {max_y}]")
                print(f"              目标边界x=[{target_min_x}, {target_max_x}], y=[{target_min_y}, {target_max_y}]")
            elif min_x > target_min_x or max_x < target_max_x or min_y > target_min_y or max_y < target_max_y:
                # 轮廓离边界太远，但我们仍然希望保持精确的1像素距离
                # 这可以通过调整位图尺寸来实现，但为了简单起见，我们只处理太靠近边界的情况
                print(f"    注意：轮廓离边界超过1像素距离，但保持原样")
            
            if needs_adjustment:
                print(f"    轮廓需要调整以保证至少1像素边界距离")
                
                # 计算需要的新尺寸
                # 首先确定轮廓需要的最小边界
                # 我们想要轮廓在位置[1,1]，所以需要计算平移量
                shift_x = target_min_x - min_x
                shift_y = target_min_y - min_y
                
                print(f"    计算平移：x方向{shift_x}像素，y方向{shift_y}像素")
                
                # 计算平移后的新边界
                new_min_x = min_x + shift_x
                new_max_x = max_x + shift_x
                new_min_y = min_y + shift_y
                new_max_y = max_y + shift_y
                
                # 计算轮廓尺寸
                contour_width = max_x - min_x + 1
                contour_height = max_y - min_y + 1
                
                # 新位图尺寸 = 轮廓尺寸 + 2（左右各1像素边界）
                new_width = contour_width + 2
                new_height = contour_height + 2
                
                print(f"    轮廓尺寸：{contour_width}×{contour_height}，新位图尺寸：{new_width}×{new_height}")
                
                # 创建新位图
                new_bitmap = np.zeros((new_height, new_width), dtype=offset_bitmap.dtype)
                
                # 将轮廓复制到新位图中（位置[1, 1]）
                # 使用更高效的方法：直接复制区域
                for y in range(contour_height):
                    for x in range(contour_width):
                        src_y = min_y + y
                        src_x = min_x + x
                        if (0 <= src_y < height and 0 <= src_x < width and
                            offset_bitmap[src_y, src_x] == 255):
                            new_bitmap[y + 1, x + 1] = 255
                
                offset_bitmap = new_bitmap
                print(f"    已创建新位图，尺寸：{new_width}×{new_height}")
                
                # 更新尺寸变量
                height, width = offset_bitmap.shape
                
                # 验证调整后的边界距离
                new_white_pixels = np.argwhere(offset_bitmap == 255)
                if len(new_white_pixels) > 0:
                    final_min_x = np.min(new_white_pixels[:, 1])
                    final_max_x = np.max(new_white_pixels[:, 1])
                    final_min_y = np.min(new_white_pixels[:, 0])
                    final_max_y = np.max(new_white_pixels[:, 0])
                    
                    print(f"    调整后轮廓边界：x=[{final_min_x}, {final_max_x}], y=[{final_min_y}, {final_max_y}]，图像尺寸：{width}×{height}")
                    
                    # 验证边界距离 - 现在应该是精确的一个像素距离
                    if final_min_x == 1 and final_max_x == width - 2 and final_min_y == 1 and final_max_y == height - 2:
                        print(f"    [OK] 边界距离验证通过：轮廓到各边都有精确的一个像素距离")
                    else:
                        print(f"    [WARN] 边界距离验证失败：期望x=[1, {width-2}], y=[1, {height-2}]，实际x=[{final_min_x}, {final_max_x}], y=[{final_min_y}, {final_max_y}]")
    
    return offset_bitmap


def generate_part_bitmaps(parts, output_dir="data/output/bitmaps", pixel_resolution=PIXEL_RESOLUTION, apply_offset=False, pg=CUTTING_GAP_PG, npy_output_dir=None):
    """
    为所有零件生成位图
    
    参数:
        parts: 零件列表
        output_dir: 输出目录（用于保存PNG图像）
        pixel_resolution: 像素精度（mm/pixel）
        apply_offset: 是否应用轮廓偏移
        pg: 切割间隙（mm），默认CUTTING_GAP_PG
        npy_output_dir: .npy文件输出目录（如果为None，则使用output_dir）
    
    返回:
        list: 位图信息列表
    """
    # 如果未指定npy输出目录，使用默认的template_matrix目录
    if npy_output_dir is None:
        npy_output_dir = "data/intermediate/template_matrix"
    if apply_offset:
        print(f"\n生成零件位图（像素精度：{pixel_resolution} mm/pixel，应用偏移：pg={pg}mm）...")
    else:
        print(f"\n生成零件位图（像素精度：{pixel_resolution} mm/pixel）...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    # 确保.npy输出目录存在
    os.makedirs(npy_output_dir, exist_ok=True)
    
    bitmaps_info = []
    
    for part_idx, part in enumerate(parts):
        print(f"  处理零件 #{part_idx}...")
        
        try:
            # 1. 渲染位图
            # 对于原始位图，传递pg参数以自动计算边框宽度（1+pg/2）
            bitmap, width_pixels, height_pixels, bounds = render_part_to_bitmap(
                part, pixel_resolution=pixel_resolution, pg=pg
            )
            
            # 2. 应用轮廓偏移（如果需要）
            if apply_offset:
                print(f"    应用轮廓偏移操作...")
                bitmap = apply_offset_to_bitmap(
                    bitmap,
                    pixel_resolution=pixel_resolution,
                    pg=pg
                    # 注意：ensure_border_distance使用默认值False，根据用户需求
                )
            
            # 3. 保存为图像文件
            image_path = os.path.join(output_dir, f"part_{part_idx:03d}.png")
            save_bitmap_as_image(bitmap, image_path)
            
            # 4. 保存为numpy矩阵文件（到template_matrix目录）
            matrix_path = os.path.join(npy_output_dir, f"part_{part_idx:03d}")
            save_bitmap_as_numpy(bitmap, matrix_path)
            
            # 5. 收集位图信息
            bitmaps_info.append({
                "part_index": part_idx,
                "bitmap": bitmap,
                "width_pixels": width_pixels,
                "height_pixels": height_pixels,
                "bounds": bounds,
                "pixel_resolution": pixel_resolution,
                "apply_offset": apply_offset,
                "pg": pg if apply_offset else None,
                "image_path": image_path,
                "matrix_path": matrix_path + ".npy"
            })
            
            print(f"    尺寸：{width_pixels}×{height_pixels} 像素，包围矩形：{bounds}")
            if apply_offset:
                white_pixels = np.sum(bitmap == 255)
                total_pixels = bitmap.size
                print(f"    偏移后白色像素比例：{white_pixels/total_pixels:.2%}")
            
        except Exception as e:
            print(f"    处理零件 #{part_idx} 时出错：{e}")
            import traceback
            traceback.print_exc()
    
    print(f"[OK] 已为 {len(bitmaps_info)} 个零件生成位图")
    return bitmaps_info


# -------------------------- 主处理函数 --------------------------
def process_dxf_file(input_file=None, output_dxf=None, output_image_all=None, output_image_outer=None,
                     apply_offset=True, pg=CUTTING_GAP_PG):
    """
    主处理函数：处理DXF文件，识别零件，创建Block，生成可视化图片
    
    参数:
        input_file: 输入DXF文件路径（默认使用DEFAULT_INPUT_FILE）
        output_dxf: 输出DXF文件路径（默认使用DEFAULT_OUTPUT_DXF）
        output_image_all: 所有零件图片路径（默认使用DEFAULT_OUTPUT_IMAGE_ALL）
        output_image_outer: 仅外轮廓图片路径（默认使用DEFAULT_OUTPUT_IMAGE_OUTER）
        apply_offset: 是否应用轮廓偏移（默认True）
        pg: 切割间隙（mm），默认4mm
    
    返回:
        bool: 是否成功
    """
    # 设置默认值
    if input_file is None:
        input_file = DEFAULT_INPUT_FILE
    if output_dxf is None:
        output_dxf = DEFAULT_OUTPUT_DXF
    if output_image_all is None:
        output_image_all = DEFAULT_OUTPUT_IMAGE_ALL
    if output_image_outer is None:
        output_image_outer = DEFAULT_OUTPUT_IMAGE_OUTER
    
    print("=" * 70)
    print("DXF Processor - 零件识别与可视化")
    print(f"输入文件: {input_file}")
    print(f"输出DXF: {output_dxf}")
    print(f"输出图片（所有零件）: {output_image_all}")
    print(f"输出图片（仅外轮廓）: {output_image_outer}")
    print("=" * 70)
    
    # 0. 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 '{input_file}' 不存在！")
        print(f"当前目录：{os.getcwd()}")
        print("请确保文件路径正确。")
        return False
    
    print(f"输入文件大小：{os.path.getsize(input_file)} 字节")
    
    # 1. 读取DXF文件
    print(f"正在读取DXF文件：{input_file}")
    try:
        doc = ezdxf.readfile(input_file)
        msp = doc.modelspace()
        print(f"成功读取DXF文件，版本：{doc.dxfversion}")
        print(f"模型空间实体数量：{len(msp)}")
    except Exception as e:
        print(f"读取DXF失败：{e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 2. 提取所有封闭轮廓
    print("正在提取封闭轮廓...")
    try:
        contours = extract_closed_contours(msp)
        if not contours:
            print("警告：未找到有效封闭轮廓！")
            print("可能的原因：")
            print("  1. 文件中没有封闭轮廓")
            print("  2. 轮廓由不支持的实体类型组成")
            print("  3. 面积阈值设置过高")
            return False
        print(f"共提取到 {len(contours)} 个封闭轮廓")
    except Exception as e:
        print(f"提取封闭轮廓时出错：{e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 按包含关系分组为零件
    print("正在分析轮廓包含关系...")
    try:
        parts = group_contours_by_part(contours)
        if not parts:
            print("警告：未识别出任何零件！")
            return False
        print(f"共识别出 {len(parts)} 个零件")
        
        # 显示零件信息
        for i, part in enumerate(parts):
            outer_area = part["outer"]["area"]
            holes_count = len(part["holes"])
            center_x, center_y = part["center"]
            print(f"  零件 #{i}: 外轮廓面积={outer_area:.2f}, 内孔数量={holes_count}, 中心点=({center_x:.2f}, {center_y:.2f})")
    except Exception as e:
        print(f"分析轮廓包含关系时出错：{e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 生成零件位图（仅外轮廓）
    print("\n正在生成零件位图（仅外轮廓）...")
    try:
        # 设置位图输出目录
        bitmap_output_dir = "data/output/bitmaps"
        
        # 根据是否应用偏移设置子目录
        if apply_offset:
            bitmap_output_dir = f"data/output/bitmaps_offset_pg{pg}"
            print(f"  应用轮廓偏移：pg={pg}mm")
        
        # 生成位图
        bitmaps_info = generate_part_bitmaps(
            parts,
            output_dir=bitmap_output_dir,
            pixel_resolution=PIXEL_RESOLUTION,
            apply_offset=apply_offset,
            pg=pg,
            npy_output_dir="data/intermediate/template_matrix"
        )
        
        if bitmaps_info:
            print(f"[OK] 已为 {len(bitmaps_info)} 个零件生成位图")
            print(f"  位图文件保存在：{bitmap_output_dir}")
            
            # 显示位图统计信息
            total_pixels = sum(info["width_pixels"] * info["height_pixels"] for info in bitmaps_info)
            avg_width = sum(info["width_pixels"] for info in bitmaps_info) / len(bitmaps_info)
            avg_height = sum(info["height_pixels"] for info in bitmaps_info) / len(bitmaps_info)
            print(f"  总像素数：{total_pixels:,}")
            print(f"  平均尺寸：{avg_width:.0f}×{avg_height:.0f} 像素")
        else:
            print("警告：未生成任何位图")
    except Exception as e:
        print(f"生成位图时出错：{e}")
        import traceback
        traceback.print_exc()
        # 位图生成失败不影响后续处理，继续执行
    
    # 5. 创建零件Block
    print("\n正在创建零件Block...")
    try:
        create_part_blocks(doc, parts)
    except Exception as e:
        print(f"创建零件Block时出错：{e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. 保存新DXF文件
    print("正在保存输出文件...")
    try:
        doc.saveas(output_dxf)
        print(f"处理完成！结果已保存至：{output_dxf}")
        print(f"输出文件大小：{os.path.getsize(output_dxf)} 字节")
    except Exception as e:
        print(f"保存DXF失败：{e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 7. 生成可视化图片
    print("\n正在生成可视化图片...")
    
    # 7.1 生成所有零件图片（包含外轮廓和内孔）
    print(f"1. 生成所有零件图片：{output_image_all}")
    try:
        success_all = create_visualization(input_file, output_image_all)
        if success_all:
            print(f"[OK] 所有零件图片已生成：{output_image_all}")
        else:
            print(f"✗ 所有零件图片生成失败")
    except Exception as e:
        print(f"生成所有零件图片时出错：{e}")
    
    # 7.2 生成仅外轮廓图片（使用有序绘制方法）
    print(f"\n2. 生成仅外轮廓图片：{output_image_outer}")
    try:
        # 创建仅包含外轮廓的零件数据（移除内孔）
        parts_outer_only = []
        for part in parts:
            parts_outer_only.append({
                "outer": part["outer"],
                "holes": [],  # 移除内孔
                "center": part["center"]
            })
        
        # 使用visualize_parts函数生成图片
        if parts_outer_only:
            success_outer = visualize_parts(parts_outer_only, output_image_outer)
            if success_outer:
                print(f"[OK] 仅外轮廓图片已生成：{output_image_outer}")
            else:
                print(f"✗ 仅外轮廓图片生成失败")
        else:
            print("警告：没有零件数据可用于生成仅外轮廓图片")
    except Exception as e:
        print(f"生成仅外轮廓图片时出错：{e}")
    
    print("\n" + "=" * 70)
    print("DXF处理完成！")
    print("=" * 70)
    return True


# ============================================================================
# 图形操作模块 - 根据CSV变换指令对零件进行复制、旋转、平移操作
# ============================================================================

def read_transformation_csv(csv_path):
    """
    读取transformation_info中的CSV文件，解析变换指令
    
    参数:
        csv_path: CSV文件路径
        
    返回:
        list: 变换指令列表，每个元素为字典，包含：
            - part_idx: 零件序号
            - copy_count: 副本个数
            - copy_idx: 副本序号
            - angle: 旋转角度（度）
            - center_x: 中心点X坐标（mm）
            - center_y: 中心点Y坐标（mm）
    """
    import csv
    
    transformations = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            print(f"CSV列名: {fieldnames}")
            
            for row in reader:
                # 尝试使用英文列名，如果失败则使用中文列名
                try:
                    # 首先尝试英文列名
                    part_idx = int(row['part_index'])
                    copy_count = int(row['copy_count'])
                    copy_idx = int(row['copy_index'])
                    angle = float(row['angle'])
                    center_x = float(row['center_x_mm'])
                    center_y = float(row['center_y_mm'])
                except KeyError:
                    # 如果英文列名不存在，尝试中文列名
                    try:
                        part_idx = int(row['零件序号'])
                        copy_count = int(row['副本个数'])
                        copy_idx = int(row['副本序号'])
                        angle = float(row['角度'])
                        center_x = float(row['中心点X坐标(mm)'])
                        center_y = float(row['中心点Y坐标(mm)'])
                    except KeyError as e:
                        print(f"错误: CSV文件中缺少必要的列: {e}")
                        print(f"可用的列: {list(row.keys())}")
                        return []
                
                transformations.append({
                    'part_idx': part_idx,
                    'copy_count': copy_count,
                    'copy_idx': copy_idx,
                    'angle': angle,
                    'center_x': center_x,
                    'center_y': center_y
                })
        
        print(f"成功读取 {len(transformations)} 条变换指令")
        return transformations
        
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def rotate_bitmap(bitmap, angle_degrees):
    """
    旋转位图矩阵
    
    参数:
        bitmap: 二维numpy数组（位图矩阵）
        angle_degrees: 旋转角度（度）
        
    返回:
        numpy.ndarray: 旋转后的位图矩阵
    """
    from scipy import ndimage
    
    # 将角度转换为弧度
    angle_rad = np.deg2rad(angle_degrees)
    
    # 使用scipy的旋转函数
    rotated = ndimage.rotate(bitmap, angle_degrees, reshape=True, order=0, mode='constant', cval=0)
    
    return rotated


def apply_transformation_to_bitmap(bitmap, angle_degrees, offset_x_pixels, offset_y_pixels, pixel_resolution):
    """
    对位图应用旋转和平移变换
    
    参数:
        bitmap: 原始位图矩阵
        angle_degrees: 旋转角度（度）
        offset_x_pixels: X方向平移像素数
        offset_y_pixels: Y方向平移像素数
        pixel_resolution: 像素精度（mm/pixel）
        
    返回:
        numpy.ndarray: 变换后的位图矩阵
    """
    # 1. 旋转位图
    if abs(angle_degrees) > 0.1:  # 如果角度不为0
        rotated_bitmap = rotate_bitmap(bitmap, angle_degrees)
    else:
        rotated_bitmap = bitmap.copy()
    
    # 2. 创建足够大的画布来容纳平移后的位图
    height, width = rotated_bitmap.shape
    
    # 计算平移后的边界
    new_height = height + abs(offset_y_pixels)
    new_width = width + abs(offset_x_pixels)
    
    # 创建新画布
    transformed_bitmap = np.zeros((new_height, new_width), dtype=rotated_bitmap.dtype)
    
    # 3. 计算平移后的位置
    start_x = max(0, offset_x_pixels) if offset_x_pixels >= 0 else 0
    start_y = max(0, offset_y_pixels) if offset_y_pixels >= 0 else 0
    
    src_start_x = max(0, -offset_x_pixels) if offset_x_pixels < 0 else 0
    src_start_y = max(0, -offset_y_pixels) if offset_y_pixels < 0 else 0
    
    # 4. 复制位图到新位置
    src_height = min(height - src_start_y, new_height - start_y)
    src_width = min(width - src_start_x, new_width - start_x)
    
    if src_height > 0 and src_width > 0:
        transformed_bitmap[start_y:start_y+src_height, start_x:start_x+src_width] = \
            rotated_bitmap[src_start_y:src_start_y+src_height, src_start_x:src_start_x+src_width]
    
    return transformed_bitmap


def create_arranged_parts_dxf(source_dxf_path, transformations, output_dxf_path):
    """
    根据CSV变换指令创建排布好的零件DXF文件
    使用直接实体变换方法（参考trial4.py）
    
    参数:
        source_dxf_path: 源DXF文件路径（包含原始零件）
        transformations: 变换指令列表
        output_dxf_path: 输出DXF文件路径
        
    返回:
        bool: 是否成功
    """
    import ezdxf
    from ezdxf.math import Matrix44
    import math
    
    try:
        # 1. 读取源DXF文件
        print(f"读取源DXF文件: {source_dxf_path}")
        source_doc = ezdxf.readfile(source_dxf_path)
        source_msp = source_doc.modelspace()
        
        # 2. 创建新的DXF文档
        new_doc = ezdxf.new('R2010')
        new_msp = new_doc.modelspace()
        
        print(f"创建排布好的DXF文件: {output_dxf_path}")
        print(f"变换指令数量: {len(transformations)}")
        
        # 3. 从源DXF文件中提取所有零件（使用与trial4.py相同的方法）
        print("正在提取零件轮廓...")
        from dxf_processor import extract_closed_contours, group_contours_by_part, get_part_entities
        
        # 提取封闭轮廓
        contours = extract_closed_contours(source_msp)
        if not contours:
            print("错误: 未找到有效封闭轮廓")
            return False
        
        # 按包含关系分组为零件
        parts = group_contours_by_part(contours)
        if not parts:
            print("错误: 未识别出任何零件")
            return False
        
        print(f"成功识别出 {len(parts)} 个零件")
        
        # 4. 为每个零件创建变换后的副本
        transformed_count = 0
        
        for i, transformation in enumerate(transformations):
            part_idx = transformation['part_idx']
            angle = transformation['angle']
            target_center_x = transformation['center_x']
            target_center_y = transformation['center_y']
            
            print(f"  处理变换 {i+1}: 零件 {part_idx}, 角度 {angle}°, 目标中心点 ({target_center_x}, {target_center_y})")
            
            # 检查零件索引是否有效
            if part_idx < 0 or part_idx >= len(parts):
                print(f"    警告: 零件索引 {part_idx} 无效，跳过")
                continue
            
            # 获取指定零件
            selected_part = parts[part_idx]
            original_center = selected_part["center"]
            
            # 获取零件的所有实体
            part_entities = get_part_entities(selected_part)
            if not part_entities:
                print(f"    警告: 零件 {part_idx} 没有实体，跳过")
                continue
            
            # 坐标系统变换：computation_core坐标系 -> DXF坐标系
            # 注意：computation_core现在输出的是左上角原点坐标（与substrate_visualization.png一致）
            # DXF坐标系也是左上角原点，x向右，y向下
            # 因此不需要进行Y坐标翻转
            
            # 基板尺寸（从computation_core导入）
            from main import SUBSTRATE_WIDTH_MM, SUBSTRATE_HEIGHT_MM
            
            # 将computation_core坐标转换为DXF坐标
            # computation_core: 原点在左上角，x向右，y向下（与substrate_visualization.png一致）
            # DXF: 原点在左上角，x向右，y向下
            
            # 因此坐标直接使用，不需要变换
            target_center_x_dxf = target_center_x
            target_center_y_dxf = target_center_y
            
            # 角度变换
            angle_dxf = angle
            
            print(f"    坐标变换: 原始({target_center_x:.1f}, {target_center_y:.1f}) -> DXF({target_center_x_dxf:.1f}, {target_center_y_dxf:.1f})")
            print(f"    角度变换: 原始{angle}° -> DXF{angle_dxf}°")
            
            # 计算平移矢量：从原始中心点到目标中心点（使用DXF坐标）
            dx = target_center_x_dxf - original_center[0]
            dy = target_center_y_dxf - original_center[1]
            translation_vector = (dx, dy)
            
            # 创建组合变换矩阵：先镜像翻转，再旋转，最后平移
            # 1. 绕水平轴（X轴）镜像翻转（Y坐标取反）
            #    以最小外包矩形中心为翻转中心：T(-center) * S(1, -1, 1) * T(center)
            # 2. 绕中心点旋转：T(-center) * Rz(angle) * T(center)
            # 3. 平移：T(dx, dy)
            
            # 1. 镜像翻转矩阵（绕水平轴，Y坐标取反）
            # 平移到原点
            translate_to_origin = Matrix44.translate(-original_center[0], -original_center[1], 0)
            # Y轴镜像翻转（缩放Y坐标为-1）
            mirror_y = Matrix44.scale(1.0, -1.0, 1.0)
            # 平移回原位置
            translate_back = Matrix44.translate(original_center[0], original_center[1], 0)
            # 完整的镜像翻转矩阵
            mirror_matrix = translate_to_origin @ mirror_y @ translate_back
            
            # 2. 旋转矩阵
            if angle_dxf != 0.0:
                angle_rad = math.radians(angle_dxf)
                # 绕中心点旋转的变换（同样以最小外包矩形中心为旋转中心）
                rotate_z = Matrix44.z_rotate(angle_rad)
                # 完整的旋转矩阵：T(-center) * Rz(angle) * T(center)
                rotation_matrix = translate_to_origin @ rotate_z @ translate_back
            else:
                # 单位矩阵
                rotation_matrix = Matrix44()
            
            # 3. 平移矩阵
            if dx != 0.0 or dy != 0.0:
                translation_matrix = Matrix44.translate(dx, dy, 0)
            else:
                translation_matrix = Matrix44()
            
            # 组合变换：先镜像翻转，再旋转，最后平移
            # 注意：矩阵乘法顺序是从右到左应用变换
            transform_matrix =   rotation_matrix @ translation_matrix
            
            print(f"    添加了水平轴镜像翻转（以最小外包矩形中心({original_center[0]:.1f}, {original_center[1]:.1f})为翻转中心）")
            
            # 对每个实体应用变换并复制到新文档
            for entity in part_entities:
                try:
                    # 复制实体
                    entity_copy = entity.copy()
                    # 应用变换
                    entity_copy.transform(transform_matrix)
                    # 添加到新文档的模型空间
                    new_msp.add_entity(entity_copy)
                    transformed_count += 1
                except Exception as e:
                    print(f"    警告: 无法变换实体 {entity.dxftype()}: {e}")
        
        # 5. 保存DXF文件
        new_doc.saveas(output_dxf_path)
        print(f"排布好的DXF文件已保存: {output_dxf_path}")
        print(f"成功变换并添加了 {transformed_count} 个实体")
        print(f"共处理了 {len(transformations)} 个零件副本")
        return True
        
    except Exception as e:
        print(f"创建排布好的DXF文件失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def apply_graphic_transformations(input_csv, source_dxf_path, output_dxf_path):
    """
    图形操作模块主函数：读取CSV变换指令并应用到零件DXF实体
    
    参数:
        input_csv: 输入CSV文件路径
        source_dxf_path: 源DXF文件路径（包含零件块）
        output_dxf_path: 输出DXF文件路径（排布好的零件图）
        
    返回:
        bool: 是否成功
    """
    print("\n" + "=" * 70)
    print("图形操作模块 - DXF实体操作")
    print("=" * 70)
    
    # 1. 读取变换指令
    transformations = read_transformation_csv(input_csv)
    if not transformations:
        print("错误: 没有读取到变换指令")
        return False
    
    # 2. 检查源DXF文件是否存在
    if not os.path.exists(source_dxf_path):
        print(f"错误: 源DXF文件不存在: {source_dxf_path}")
        print(f"请先运行DXF处理流程生成零件块")
        return False
    
    # 3. 创建排布好的DXF文件
    success = create_arranged_parts_dxf(source_dxf_path, transformations, output_dxf_path)
    
    if success:
        print("\n图形操作完成！")
        print(f"排布好的零件图已保存: {output_dxf_path}")
    else:
        print("\n图形操作失败！")
    
    return success


def main():
    """主函数：命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DXF Processor - 零件识别与可视化')
    parser.add_argument('-i', '--input', default=DEFAULT_INPUT_FILE,
                       help=f'输入DXF文件路径 (默认: {DEFAULT_INPUT_FILE})')
    parser.add_argument('-o', '--output-dxf', default=DEFAULT_OUTPUT_DXF,
                       help=f'输出DXF文件路径 (默认: {DEFAULT_OUTPUT_DXF})')
    parser.add_argument('--image-all', default=DEFAULT_OUTPUT_IMAGE_ALL,
                       help=f'所有零件图片路径 (默认: {DEFAULT_OUTPUT_IMAGE_ALL})')
    parser.add_argument('--image-outer', default=DEFAULT_OUTPUT_IMAGE_OUTER,
                       help=f'仅外轮廓图片路径 (默认: {DEFAULT_OUTPUT_IMAGE_OUTER})')
    parser.add_argument('--apply-offset', action='store_true', default=True,
                       help='应用轮廓偏移（用于零件排布计算），默认启用')
    parser.add_argument('--no-offset', action='store_false', dest='apply_offset',
                       help='禁用轮廓偏移')
    parser.add_argument('--pg', type=float, default=CUTTING_GAP_PG,
                       help=f'切割间隙（mm），默认{CUTTING_GAP_PG}mm')
    parser.add_argument('--visualize-only', action='store_true',
                       help='仅生成可视化图片，不创建DXF文件')
    parser.add_argument('--ordered', action='store_true',
                       help='使用有序绘制方法生成可视化图片')
    
    args = parser.parse_args()
    
    if args.visualize_only:
        # 仅生成可视化图片
        print("仅生成可视化图片模式...")
        if args.ordered:
            success = create_ordered_visualization(args.input, args.image_all)
        else:
            success = create_visualization(args.input, args.image_all)
        
        if success:
            print("\n可视化图片生成完成！")
            return 0
        else:
            print("\n可视化图片生成失败！")
            return 1
    else:
        # 完整处理流程
        success = process_dxf_file(
            input_file=args.input,
            output_dxf=args.output_dxf,
            output_image_all=args.image_all,
            output_image_outer=args.image_outer,
            apply_offset=args.apply_offset,
            pg=args.pg
        )
        
        if success:
            return 0
        else:
            return 1

if __name__ == "__main__":
    sys.exit(main())
