#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trial4.py - 零件变换工具
功能：用户指定零件序号、旋转角度和移动位置矢量，对指定零件进行变换
调用trial2.py与trial3.py中的函数实现功能
"""

import ezdxf
import math
import os
import sys
from ezdxf.math import Matrix44

# 检查必要的依赖
try:
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    SHAPELY_AVAILABLE = True
except ImportError:
    print("错误：未找到shapely模块！")
    print("请安装shapely模块：pip install shapely")
    SHAPELY_AVAILABLE = False
    sys.exit(1)

# 从trial3.py中复制必要的函数和配置
SAMPLING_STEP = 0.1                # 样条线离散化步长
AREA_THRESHOLD = 0.01              # 过滤微小无效轮廓的面积阈值

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

def points_equal(p1, p2, tolerance=0.01):
    """Check if two points are equal (within tolerance range)"""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5 < tolerance

def find_connected_chains(entities):
    """查找首尾相连的实体链"""
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
        
        # 处理样条线（复杂外轮廓）
        elif entity_type == "SPLINE":
            # 检查样条线是否封闭 - 使用 closed 属性而不是 is_closed
            if not hasattr(entity, 'closed') or not entity.closed:
                continue
            # 简化处理：使用控制点
            if entity.fit_points:
                points = entity.fit_points
            else:
                points = entity.control_points
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
        if poly and poly.area > AREA_THRESHOLD:
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
                # 简化处理：只添加起点和终点
                endpoints = get_entity_endpoints(entity)
                if endpoints:
                    start, end = endpoints
                    
                    if not points:
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
                # 简化处理：使用控制点
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
        
        # 创建多边形
        try:
            if len(points) >= 3:
                poly = Polygon(points)
                poly = make_valid(poly) if not poly.is_valid else poly
                if poly and poly.area > AREA_THRESHOLD:
                    closed_contours.append({
                        "entity": chain[0],  # 使用链中的第一个实体作为代表
                        "polygon": poly,
                        "area": poly.area,
                        "type": "chain",
                        "chain": chain  # 保存整个链供后续使用
                    })
        except:
            pass
    
    print(f"总共找到 {len(closed_contours)} 个封闭轮廓")
    
    # 按面积降序排序（大轮廓优先作为外轮廓）
    closed_contours.sort(key=lambda x: x["area"], reverse=True)
    return closed_contours

def group_contours_by_part(contours):
    """根据包含关系分组轮廓（外轮廓+内孔=一个零件）"""
    if not SHAPELY_AVAILABLE:
        print("错误：shapely模块不可用！")
        return []
    
    parts = []
    used_contours = set()  # 标记已分配的轮廓
    
    for idx, outer_candidate in enumerate(contours):
        if idx in used_contours:
            continue
        
        outer_poly = outer_candidate["polygon"]
        # 判断是否为外轮廓：不被任何更大的轮廓包含
        is_outer = True
        for j in range(idx):
            if j in used_contours:
                continue
            if contours[j]["polygon"].contains(outer_poly):
                is_outer = False
                break
        
        if not is_outer:
            continue
        
        # 收集该外轮廓包含的所有内孔
        holes = []
        for j, hole_candidate in enumerate(contours):
            if j == idx or j in used_contours:
                continue
            
            hole_poly = hole_candidate["polygon"]
            if outer_poly.contains(hole_poly):
                holes.append(hole_candidate)
                used_contours.add(j)
        
        # 封装零件数据
        parts.append({
            "outer": outer_candidate,
            "holes": holes,
            "center": (outer_poly.centroid.x, outer_poly.centroid.y),  # 零件中心点
            "index": len(parts)  # 添加索引号
        })
        used_contours.add(idx)
    
    return parts

def get_part_entities(part):
    """获取零件中的所有实体（外轮廓和内孔）"""
    entities = []
    
    # 添加外轮廓实体
    outer_type = part["outer"].get("type", "single")
    if outer_type == "chain":
        entities.extend(part["outer"].get("chain", []))
    else:
        entities.append(part["outer"]["entity"])
    
    # 添加内孔实体
    for hole in part["holes"]:
        hole_type = hole.get("type", "single")
        if hole_type == "chain":
            entities.extend(hole.get("chain", []))
        else:
            entities.append(hole["entity"])
    
    return entities

def rotate_part_entities(entities, center, angle_degrees):
    """
    将实体列表绕通过指定中心点且平行z轴的轴旋转
    
    参数:
        entities: 实体列表
        center: 旋转中心点 (x, y)
        angle_degrees: 旋转角度（度），正角度为逆时针
    """
    import math
    
    # 将角度转换为弧度
    angle_rad = math.radians(angle_degrees)
    
    # 创建绕z轴旋转的变换矩阵
    # 绕任意点旋转的变换：T(-center) * Rz(angle) * T(center)
    
    # 1. 平移到原点
    translate_to_origin = Matrix44.translate(-center[0], -center[1], 0)
    
    # 2. 绕z轴旋转
    rotate_z = Matrix44.z_rotate(angle_rad)
    
    # 3. 平移回原位置
    translate_back = Matrix44.translate(center[0], center[1], 0)
    
    # 组合变换矩阵
    transform_matrix = translate_to_origin @ rotate_z @ translate_back
    
    # 对每个实体应用旋转变换
    rotated_count = 0
    for entity in entities:
        try:
            entity.transform(transform_matrix)
            rotated_count += 1
        except Exception as e:
            print(f"警告：无法旋转实体 {entity.dxftype()}：{e}")
    
    return rotated_count

def translate_part_entities(entities, dx=0.0, dy=0.0):
    """将实体列表沿指定方向平移"""
    # 创建平移矩阵
    translate_matrix = Matrix44.translate(dx, dy, 0)
    
    # 对每个实体应用平移变换
    translated_count = 0
    for entity in entities:
        try:
            entity.transform(translate_matrix)
            translated_count += 1
        except Exception as e:
            print(f"警告：无法平移实体 {entity.dxftype()}：{e}")
    
    return translated_count

def transform_part_entities(entities, center, rotation_angle=0.0, translation_vector=(0.0, 0.0)):
    """
    对零件实体应用完整的变换：先旋转后平移
    
    参数:
        entities: 实体列表
        center: 旋转中心点 (x, y)
        rotation_angle: 旋转角度（度），正角度为逆时针
        translation_vector: 平移矢量 (dx, dy)
    """
    import math
    
    dx, dy = translation_vector
    
    # 如果不需要变换，直接返回
    if rotation_angle == 0.0 and dx == 0.0 and dy == 0.0:
        return 0
    
    # 创建组合变换矩阵：先旋转后平移
    # 绕中心点旋转：T(-center) * Rz(angle) * T(center)
    # 然后平移：T(dx, dy)
    
    if rotation_angle != 0.0:
        angle_rad = math.radians(rotation_angle)
        
        # 绕中心点旋转的变换
        translate_to_origin = Matrix44.translate(-center[0], -center[1], 0)
        rotate_z = Matrix44.z_rotate(angle_rad)
        translate_back = Matrix44.translate(center[0], center[1], 0)
        
        # 旋转矩阵
        rotation_matrix = translate_to_origin @ rotate_z @ translate_back
    else:
        # 单位矩阵
        rotation_matrix = Matrix44()
    
    # 平移矩阵
    if dx != 0.0 or dy != 0.0:
        translation_matrix = Matrix44.translate(dx, dy, 0)
    else:
        translation_matrix = Matrix44()
    
    # 组合变换：先旋转后平移
    transform_matrix = rotation_matrix @ translation_matrix
    
    # 对每个实体应用变换
    transformed_count = 0
    for entity in entities:
        try:
            entity.transform(transform_matrix)
            transformed_count += 1
        except Exception as e:
            print(f"警告：无法变换实体 {entity.dxftype()}：{e}")
    
    return transformed_count

def transform_part_with_parameters(part_index=0, rotation_angle=0.0, target_center=None, 
                                  input_file="completeINPUT.dxf", 
                                  output_file="completeOUTPUT_transformed.dxf"):
    """
    根据参数变换指定零件
    
    参数:
        part_index: 待操作零件的序号（0-based索引）
        rotation_angle: 旋转角度（度），正角度为逆时针
        target_center: 目标位置中心点 (x, y)，如果为None则不移动
        input_file: 输入DXF文件路径
        output_file: 输出DXF文件路径
    """
    print("=" * 70)
    print("trial4.py - 零件变换工具")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"零件序号: {part_index}")
    print(f"旋转角度: {rotation_angle}° ({'逆时针' if rotation_angle >= 0 else '顺时针'})")
    if target_center:
        print(f"目标位置: ({target_center[0]:.2f}, {target_center[1]:.2f})")
    else:
        print(f"目标位置: 不移动")
    print("=" * 70)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 '{input_file}' 不存在！")
        print(f"当前目录：{os.getcwd()}")
        return False
    
    print(f"输入文件大小：{os.path.getsize(input_file)} 字节")
    
    # 1. 读取DXF文件
    print(f"\n正在读取DXF文件：{input_file}")
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
    print("\n正在提取封闭轮廓...")
    try:
        contours = extract_closed_contours(msp)
        if not contours:
            print("错误：未找到有效封闭轮廓！")
            return False
        print(f"共提取到 {len(contours)} 个封闭轮廓")
    except Exception as e:
        print(f"提取封闭轮廓时出错：{e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 按包含关系分组为零件
    print("\n正在分析轮廓包含关系...")
    try:
        parts = group_contours_by_part(contours)
        if not parts:
            print("错误：未识别出任何零件！")
            return False
        print(f"共识别出 {len(parts)} 个零件")
        
        # 显示零件信息
        for i, part in enumerate(parts):
            outer_type = part["outer"].get("type", "single")
            hole_count = len(part["holes"])
            center = part["center"]
            print(f"  零件 #{i}: 外轮廓类型={outer_type}, 内孔数={hole_count}, 中心点=({center[0]:.2f}, {center[1]:.2f})")
    except Exception as e:
        print(f"分析轮廓包含关系时出错：{e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 检查零件索引是否有效
    if part_index < 0 or part_index >= len(parts):
        print(f"错误：零件索引 {part_index} 无效！")
        print(f"有效索引范围: 0 到 {len(parts)-1}")
        return False
    
    # 5. 获取指定零件
    selected_part = parts[part_index]
    original_center = selected_part["center"]
    print(f"\n选择零件 #{part_index}:")
    print(f"  外轮廓类型：{selected_part['outer'].get('type', 'single')}")
    print(f"  内孔数量：{len(selected_part['holes'])}")
    print(f"  原始中心点：({original_center[0]:.2f}, {original_center[1]:.2f})")
    
    # 6. 获取该零件的所有实体
    part_entities = get_part_entities(selected_part)
    print(f"  该零件包含 {len(part_entities)} 个实体")
    
    # 7. 计算平移矢量
    if target_center:
        # 位置矢量：从原始中心点到目标中心点
        dx = target_center[0] - original_center[0]
        dy = target_center[1] - original_center[1]
        translation_vector = (dx, dy)
        distance = (dx*dx + dy*dy)**0.5  # 使用幂运算代替math.sqrt
        print(f"\n位置矢量计算:")
        print(f"  原始中心点: ({original_center[0]:.2f}, {original_center[1]:.2f})")
        print(f"  目标中心点: ({target_center[0]:.2f}, {target_center[1]:.2f})")
        print(f"  平移矢量: ({dx:.2f}, {dy:.2f})")
        print(f"  移动距离: {distance:.2f} 单位")
    else:
        translation_vector = (0.0, 0.0)
        print(f"\n位置矢量: 不移动 (0, 0)")
    
    # 8. 应用变换：先旋转后平移
    print(f"\n正在应用变换...")
    print(f"  1. 绕中心点旋转 {rotation_angle}°")
    print(f"  2. 沿矢量 ({translation_vector[0]:.2f}, {translation_vector[1]:.2f}) 平移")
    
    transformed_count = transform_part_entities(
        part_entities,
        center=original_center,
        rotation_angle=rotation_angle,
        translation_vector=translation_vector
    )
    
    print(f"  成功变换 {transformed_count} 个实体")
    
    # 9. 计算变换后的中心点（用于验证）
    if rotation_angle != 0.0 or translation_vector != (0.0, 0.0):
        # 简单计算：旋转后平移
        angle_rad = math.radians(rotation_angle)
        # 绕原点旋转
        rotated_x = original_center[0] * math.cos(angle_rad) - original_center[1] * math.sin(angle_rad)
        rotated_y = original_center[0] * math.sin(angle_rad) + original_center[1] * math.cos(angle_rad)
        # 平移
        final_x = rotated_x + translation_vector[0]
        final_y = rotated_y + translation_vector[1]
        
        print(f"\n变换验证:")
        print(f"  原始中心点: ({original_center[0]:.2f}, {original_center[1]:.2f})")
        print(f"  旋转后中心点: ({rotated_x:.2f}, {rotated_y:.2f})")
        print(f"  最终中心点: ({final_x:.2f}, {final_y:.2f})")
    
    # 10. 保存修改后的DXF文件
    print(f"\n正在保存输出文件：{output_file}")
    try:
        doc.saveas(output_file)
        print(f"处理完成！结果已保存至：{output_file}")
        print(f"输出文件大小：{os.path.getsize(output_file)} 字节")
        return True
    except Exception as e:
        print(f"保存DXF失败：{e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function: Demonstrate how to use transformation functionality"""
    
    # User parameter configuration area
    # ============================================
    # Please modify the following parameters as needed:
    
    # 1. Part index to operate on (0-based index)
    PART_INDEX = 0
    
    # 2. Rotation angle (degrees), positive for counterclockwise, negative for clockwise
    ROTATION_ANGLE = -30.0  # Rotate 30 degrees counterclockwise
    
    # 3. Target position center point (x, y)
    #    If set to None, the part will only rotate without moving
    #    Current position is the origin of the position vector, target position is the endpoint
    TARGET_CENTER = (1000.0, 500.0)  # Example target position in global coordinate system
    
    # 4. Input and output files
    INPUT_FILE = "completeINPUT.dxf"
    OUTPUT_FILE = "completeOUTPUT_transformed.dxf"
    
    # 5. Visualization options
    VISUALIZE_PARTS = True  # Whether to generate part visualization image
    VISUALIZATION_FILE = "parts_visualization.png"  # Visualization image filename
    USE_ORDERED_DRAWING = True  # Whether to use ordered drawing method (solves missing holes issue)
    
    # ============================================
    
    print("=" * 70)
    print("trial4.py - Part Transformation Tool")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Part index: {PART_INDEX}")
    print(f"Rotation angle: {ROTATION_ANGLE}°")
    print(f"Target position: {TARGET_CENTER}")
    print("=" * 70)
    
    # Step 1: Generate part visualization (if enabled)
    if VISUALIZE_PARTS:
        print("\nStep 1: Generating part visualization image...")
        try:
            # Import trial5 module
            import trial5
            
            # Choose drawing method based on configuration
            if USE_ORDERED_DRAWING:
                print("Using ordered drawing method (solves missing holes issue)...")
                visualization_success = trial5.create_ordered_visualization(
                    INPUT_FILE,
                    VISUALIZATION_FILE
                )
            else:
                print("Using original drawing method...")
                visualization_success = trial5.create_visualization(
                    INPUT_FILE,
                    VISUALIZATION_FILE
                )
            
            if visualization_success:
                print(f"Part visualization image generated: {VISUALIZATION_FILE}")
                print("Please check the image to confirm part index and structure.")
                if USE_ORDERED_DRAWING:
                    print("Note: Ordered drawing method used - better for parts with connection issues.")
            else:
                print("Warning: Part visualization generation failed, but will continue with transformation.")
        except ImportError as e:
            print(f"Warning: Cannot import trial5 module, skipping visualization step. Error: {e}")
        except Exception as e:
            print(f"Warning: Error generating visualization: {e}")
    
    # Step 2: Execute transformation
    print("\nStep 2: Executing part transformation...")
    success = transform_part_with_parameters(
        part_index=PART_INDEX,
        rotation_angle=ROTATION_ANGLE,
        target_center=TARGET_CENTER,
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE
    )
    
    if success:
        print("\n" + "=" * 70)
        print("Transformation completed successfully!")
        if VISUALIZE_PARTS:
            print(f"Part visualization image: {VISUALIZATION_FILE}")
        print(f"Transformation result file: {OUTPUT_FILE}")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("Transformation failed!")
        print("=" * 70)

if __name__ == "__main__":
    main()