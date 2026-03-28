import ezdxf
import math
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid
import warnings
warnings.filterwarnings("ignore")  # 忽略shapely的非关键警告

# -------------------------- 核心配置 --------------------------
DXF_INPUT_FILE = "completeINPUT.dxf"       # 输入DXF文件路径
DXF_OUTPUT_FILE = "completeOUTPUT.dxf"  # 输出DXF文件路径
PART_BLOCK_PREFIX = "PART_"        # 零件块名称前缀
SAMPLING_STEP = 0.1                # 样条线离散化步长（越小越精确，越慢）
AREA_THRESHOLD = 0.01              # 过滤微小无效轮廓的面积阈值

# -------------------------- 工具函数 --------------------------
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
        import math
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
            
            import math
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
    # 检查样条线是否封闭 - 使用 closed 属性而不是 is_closed
    if not hasattr(spline, 'closed') or not spline.closed:
        return None
    
    # 获取样条线的拟合点/控制点
    if spline.fit_points:
        points = spline.fit_points
    else:
        points = spline.control_points
    
    # 如果点数不足，进行插值采样
    if len(points) < 4:
        num_segments = max(10, int(spline.length / step))
        points = [spline.point_at(t) for t in [i/num_segments for i in range(num_segments+1)]]
    
    # 转换为shapely多边形（确保有效）
    try:
        poly = Polygon([(p.x, p.y) for p in points])
        return make_valid(poly) if not poly.is_valid else poly
    except:
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

def extract_closed_contours(msp):
    """从模型空间提取所有封闭轮廓（外轮廓/内孔）"""
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
            poly = lwpolyline_to_polygon(entity)
        
        # 处理样条线（复杂外轮廓）
        elif entity_type == "SPLINE":
            poly = spline_to_polygon(entity)
        
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
            "center": (outer_poly.centroid.x, outer_poly.centroid.y)  # 零件中心点
        })
        used_contours.add(idx)
    
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

# -------------------------- 主流程 --------------------------
def main():
    import os
    
    # 0. 检查输入文件是否存在
    if not os.path.exists(DXF_INPUT_FILE):
        print(f"错误：输入文件 '{DXF_INPUT_FILE}' 不存在！")
        print(f"当前目录：{os.getcwd()}")
        print("请确保文件路径正确。")
        return
    
    print(f"输入文件大小：{os.path.getsize(DXF_INPUT_FILE)} 字节")
    
    # 1. 读取DXF文件
    print(f"正在读取DXF文件：{DXF_INPUT_FILE}")
    try:
        doc = ezdxf.readfile(DXF_INPUT_FILE)
        msp = doc.modelspace()
        print(f"成功读取DXF文件，版本：{doc.dxfversion}")
        print(f"模型空间实体数量：{len(msp)}")
    except Exception as e:
        print(f"读取DXF失败：{e}")
        import traceback
        traceback.print_exc()
        return
    
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
            return
        print(f"共提取到 {len(contours)} 个封闭轮廓")
    except Exception as e:
        print(f"提取封闭轮廓时出错：{e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 按包含关系分组为零件
    print("正在分析轮廓包含关系...")
    try:
        parts = group_contours_by_part(contours)
        if not parts:
            print("警告：未识别出任何零件！")
            return
        print(f"共识别出 {len(parts)} 个零件")
    except Exception as e:
        print(f"分析轮廓包含关系时出错：{e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 创建零件Block
    print("正在创建零件Block...")
    try:
        create_part_blocks(doc, parts)
    except Exception as e:
        print(f"创建零件Block时出错：{e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 保存新DXF文件
    print("正在保存输出文件...")
    try:
        doc.saveas(DXF_OUTPUT_FILE)
        print(f"处理完成！结果已保存至：{DXF_OUTPUT_FILE}")
        print(f"输出文件大小：{os.path.getsize(DXF_OUTPUT_FILE)} 字节")
    except Exception as e:
        print(f"保存DXF失败：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()