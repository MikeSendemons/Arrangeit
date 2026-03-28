#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
computation_core.py - 计算核心模块

参考文献
Meng, L., Ding, L., Pu, Y. et al.
Optimizing 2D irregular packing via image processing and computational intelligence. Sci Rep 15, 12320 (2025).
https://doi.org/10.1038/s41598-025-97202-0
"""

import numpy as np
import os
import glob

from main import PIXEL_RESOLUTION, SUBSTRATE_WIDTH_MM, SUBSTRATE_HEIGHT_MM

'''
# 基板尺寸配置（毫米）
SUBSTRATE_WIDTH_MM = 180.0    # 基板宽度 (mm)
SUBSTRATE_HEIGHT_MM = 100.0    # 基板高度 (mm)

# 像素精度配置
PIXEL_RESOLUTION = 0.5         # 像素精度 (mm/pixel)
'''

def compute_correlation_matrix(template, substrate,):
    """
    计算模板与基板之间的归一化互相关矩阵
    
    参数：
    template: 二维numpy数组，模板矩阵
    substrate: 二维numpy数组，剩余基板矩阵
    
    返回：
    correlation_matrix: 二维numpy数组，尺寸为 (P-m+1, Q-n+1)
                        其中P,Q是substrate的尺寸，m,n是template的尺寸
    """
    # 获取尺寸
    m, n = template.shape
    P, Q = substrate.shape
    
    # 检查模板是否大于基板
    if m > P or n > Q:
        # 返回空矩阵表示无法计算
        return np.array([]).reshape(0, 0)
    
    # 计算有效位移范围
    u_range = P - m + 1
    v_range = Q - n + 1
    
    # 初始化相关性矩阵
    correlation_matrix = np.zeros((u_range, v_range))
    
    # 计算模板均值
    t_mean = np.mean(template)
    t_centered = template - t_mean
    
    # 对于每个位移位置计算NCC
    for u in range(u_range):
        for v in range(v_range):
            # 提取重叠区域
            substrate_patch = substrate[u:u+m, v:v+n]
            
            # 计算重叠区域均值
            s_mean = np.mean(substrate_patch)
            s_centered = substrate_patch - s_mean
            
            # 计算分子和分母
            numerator = np.sum(s_centered * t_centered)
            denominator_1 = np.sum(s_centered ** 2)
            denominator_2 = np.sum(t_centered ** 2)
            
            # 计算NCC值
            if denominator_1 != 0 and denominator_2 != 0:
                ncc_value = numerator / np.sqrt(denominator_1 * denominator_2)
            else:
                ncc_value = 0
            
            correlation_matrix[u, v] = ncc_value
    
    return correlation_matrix


def compute_ncc_fft(substrate, template, t_mean):
    """
    优化的归一化互相关计算（内存友好版）
    
    参数：
    substrate: 二维numpy数组，基板矩阵（已翻转：1=空白，0=已占用）
    template: 二维numpy数组，旋转后的模板矩阵
    t_mean: 模板均值
    
    返回：
    ncc_matrix: NCC矩阵，尺寸为 (s_h - t_h + 1) × (s_w - t_w + 1)
    """
    s_h, s_w = substrate.shape
    t_h, t_w = template.shape
    
    # 1. 模板中心化（减去均值）
    t_centered = template - t_mean
    t_var = np.sum(t_centered ** 2)
    
    # 2. 初始化NCC矩阵
    ncc_matrix = np.zeros((s_h - t_h + 1, s_w - t_w + 1))
    
    # 3. 使用更节省内存的方式计算NCC
    # 对于大尺寸模板，使用传统循环但优化计算
    for i in range(s_h - t_h + 1):
        for j in range(s_w - t_w + 1):
            # 提取子窗口
            window = substrate[i:i+t_h, j:j+t_w]
            
            # 计算窗口均值
            w_mean = np.mean(window)
            w_centered = window - w_mean
            
            # 计算协方差和
            numerator = np.sum(w_centered * t_centered)
            
            # 计算窗口方差
            w_var = np.sum(w_centered ** 2)
            
            # 计算分母
            denominator = np.sqrt(w_var * t_var)
            if denominator == 0:
                denominator = 1e-8
            
            # 计算NCC值
            ncc_matrix[i, j] = numerator / denominator
    
    return ncc_matrix


def arrange_single_template(template_idx, template_matrix, target_angle, substrate_matrix, unplaced_templates):
    """
    单个模板完整排布流程（10步详解）
    
    参数：
    template_idx: 模板索引
    template_matrix: 原始模板矩阵
    target_angle: 目标旋转角度
    substrate_matrix: 当前基板矩阵（0=未占用，1=已占用）
    unplaced_templates: 未放置模板列表（用于记录失败情况）
    
    返回：
    updated_substrate: 更新后的基板矩阵
    placement_info: 放置信息字典，包含所有排布结果
    success: 是否成功放置
    """
    # 步骤1：模板基础信息加载和校验
    if template_matrix.size == 0:  # 空模板
        print(f"模板{template_idx}：空矩阵，跳过排布")
        unplaced_templates.append(template_idx)
        return substrate_matrix, None, False
    if np.all(template_matrix == 0):  # 全空白模板
        print(f"模板{template_idx}：无有效区域，跳过排布")
        unplaced_templates.append(template_idx)
        return substrate_matrix, None, False
    
    t_h, t_w = template_matrix.shape  # 记录原始尺寸
    
    # 检查模板是否大于基板
    sub_h, sub_w = substrate_matrix.shape
    if t_h > sub_h or t_w > sub_w:
        print(f"模板{template_idx}：尺寸{t_h}×{t_w}大于基板{sub_h}×{sub_w}，跳过排布")
        unplaced_templates.append(template_idx)
        return substrate_matrix, None, False
    
    # 步骤2：模板已经由actual_data_In预先旋转，直接使用
    # 注意：template_matrix已经是旋转后的模板，不需要再次旋转
    rotated_template = template_matrix  # 直接使用传入的模板
    rt_h, rt_w = rotated_template.shape  # 模板尺寸
    
    # 步骤3：基板预处理（二值翻转）
    substrate_inv = 1 - substrate_matrix  # 翻转：1=空白，0=已占用
    
    # 步骤4：计算模板均值（固定值）
    t_mean = np.mean(rotated_template)
    if t_mean == 0:
        t_mean = 1e-8  # 避免均值为0导致分母异常
    
    # 步骤5：计算归一化互相关（NCC）矩阵
    # 使用现有的compute_correlation_matrix函数，确保正确性
    ncc_matrix = compute_correlation_matrix(rotated_template, substrate_inv)
    
    # 如果返回空矩阵（模板大于基板），跳过
    if ncc_matrix.size == 0:
        print(f"模板{template_idx}：模板尺寸大于基板，跳过排布")
        unplaced_templates.append(template_idx)
        return substrate_matrix, None, False
    
    # 步骤6：筛选有效匹配位置
    # 对于空白基板，NCC可能全为0，需要特殊处理
    if np.all(ncc_matrix == 0):
        # 如果NCC矩阵全为0，可能是空白基板，允许在有效位置放置
        # 生成所有可能位置（按行列顺序）
        valid_positions = []
        for u in range(ncc_matrix.shape[0]):
            for v in range(ncc_matrix.shape[1]):
                valid_positions.append((u, v, 0.0))  # 匹配度为0
        valid_positions_sorted = valid_positions  # 不需要排序
        print(f"模板{template_idx}：NCC矩阵全为0，使用所有{len(valid_positions_sorted)}个可能位置")
    else:
        # 正常情况：按阈值筛选
        MATCH_THRESHOLD = 0.01  # 进一步降低阈值，以包含更多位置
        u_list, v_list = np.where(ncc_matrix >= MATCH_THRESHOLD)
        valid_positions = [
            (u, v, ncc_matrix[u, v])  # (行坐标, 列坐标, 匹配度)
            for u, v in zip(u_list, v_list)
        ]
        valid_positions_sorted = sorted(valid_positions, key=lambda x: x[2], reverse=True)
        
        # 调试信息
        if len(valid_positions_sorted) == 0 and ncc_matrix.size > 0:
            print(f"模板{template_idx}：NCC矩阵范围 [{ncc_matrix.min():.3f}, {ncc_matrix.max():.3f}]，无≥{MATCH_THRESHOLD}的位置")
        elif len(valid_positions_sorted) > 0:
            print(f"模板{template_idx}：找到{len(valid_positions_sorted)}个有效位置，最佳匹配度{valid_positions_sorted[0][2]:.3f}")
    
    if len(valid_positions_sorted) == 0:
        print(f"模板{template_idx}：无符合阈值的位置，跳过排布")
        unplaced_templates.append(template_idx)
        return substrate_matrix, None, False
    
    # 步骤7：逐个验证位置（边界+重叠）
    best_pos = None
    sub_h, sub_w = substrate_matrix.shape
    
    # 首先尝试高NCC值的位置
    for (u, v, ncc_val) in valid_positions_sorted:
        # 验证1：模板完全在基板内
        if u + rt_h > sub_h or v + rt_w > sub_w:
            continue  # 越界，跳过
        # 验证2：模板的有效区域（值为1的像素）未被占用
        substrate_patch = substrate_matrix[u:u+rt_h, v:v+rt_w]
        # 只检查模板中值为1的像素对应的位置是否在基板中已被占用
        # 使用逐元素乘法：模板中为1且基板中也为1的位置表示重叠
        overlap_mask = (rotated_template == 1) & (substrate_patch == 1)
        if np.any(overlap_mask):
            continue  # 重叠，跳过
        # 找到有效位置
        best_pos = (u, v, ncc_val)
        break
    
    # 如果所有超过阈值的位置都重叠，尝试寻找任何不重叠的位置
    if best_pos is None and not np.all(ncc_matrix == 0):
        print(f"模板{template_idx}：所有超过阈值的位置都重叠，尝试寻找任何不重叠的位置...")
        # 生成所有可能位置（不按NCC值筛选）
        for u in range(ncc_matrix.shape[0]):
            for v in range(ncc_matrix.shape[1]):
                # 验证边界
                if u + rt_h > sub_h or v + rt_w > sub_w:
                    continue
                # 验证重叠：只检查模板中值为1的像素
                substrate_patch = substrate_matrix[u:u+rt_h, v:v+rt_w]
                overlap_mask = (rotated_template == 1) & (substrate_patch == 1)
                if np.any(overlap_mask):
                    continue
                # 找到有效位置（即使NCC值很低）
                best_pos = (u, v, ncc_matrix[u, v])
                print(f"模板{template_idx}：找到不重叠的位置({u},{v})，NCC值{ncc_matrix[u,v]:.6f}")
                break
            if best_pos is not None:
                break
    
    if best_pos is None:
        print(f"模板{template_idx}：无可用位置，跳过排布")
        unplaced_templates.append(template_idx)
        return substrate_matrix, None, False
    
    u, v, ncc_val = best_pos
    
    # 步骤8：更新基板占用状态
    updated_substrate = substrate_matrix.copy()
    # 只将模板中值为1的像素设置为1，保持其他像素不变
    # 使用逻辑或操作：基板区域 = 基板区域 OR 模板
    substrate_patch = updated_substrate[u:u+rt_h, v:v+rt_w]
    # 将模板中值为1的像素设置为1
    mask = (rotated_template == 1)
    substrate_patch[mask] = 1
    updated_substrate[u:u+rt_h, v:v+rt_w] = substrate_patch
    
    # 步骤9：坐标转换（像素→物理尺寸）并计算中心点
    # 左上角坐标（像素）
    left_x_pixels = v
    top_y_pixels = u
    
    # 转换为物理尺寸（毫米）
    left_x_mm = left_x_pixels * PIXEL_RESOLUTION
    top_y_mm = top_y_pixels * PIXEL_RESOLUTION
    
    # 计算模板尺寸（毫米）
    template_width_mm = rt_w * PIXEL_RESOLUTION
    template_height_mm = rt_h * PIXEL_RESOLUTION
    
    # 计算矩形模板中心点坐标（毫米）- 使用左上角原点坐标系
    # 当前计算的是左下角原点坐标系的中心点，需要转换为左上角原点坐标系
    # 转换公式：y_top = SUBSTRATE_HEIGHT_MM - y_bottom
    center_x_mm = left_x_mm + template_width_mm / 2.0
    center_y_mm_bottom = top_y_mm + template_height_mm / 2.0  # 左下角原点坐标系的y坐标
    center_y_mm_top = SUBSTRATE_HEIGHT_MM - center_y_mm_bottom  # 转换为左上角原点坐标系
    center_pos = (center_x_mm, center_y_mm_top)
    
    # 步骤10：记录排布结果
    # 将左上角坐标也转换为左上角原点坐标系
    top_y_mm_top = SUBSTRATE_HEIGHT_MM - top_y_mm  # 转换为左上角原点坐标系
    
    placement_info = {
        "template_idx": template_idx,
        "pixel_position": (u, v),
        "physical_position": center_pos,  # 存储中心点坐标（已转换为左上角原点）
        "rotation_angle": target_angle,
        "ncc_value": ncc_val,
        "template_size": (rt_h, rt_w),
        "template_size_mm": (template_width_mm, template_height_mm),
        "corner_position": (left_x_mm, top_y_mm_top)  # 保留左上角坐标（已转换为左上角原点）
    }
    
    print(f"模板{template_idx}：排布完成，位置({u},{v})，匹配度{ncc_val:.3f}")
    return updated_substrate, placement_info, True


### 1.数据准备
# 创建一个空列表data_In存储后续数据
data_In = []

# 从template_matrix文件夹中读取所有的.npy格式文件，按文件名升序排列
template_dir = "data/intermediate/template_matrix"
npy_files = glob.glob(os.path.join(template_dir, "*.npy"))
npy_files.sort()  # 按文件名升序排列

# 将.npy文件数据存到data_In的第一行
if npy_files:
    # 读取所有.npy文件并存储到data_In的第一行
    loaded_data = []
    for file_path in npy_files:
        data = np.load(file_path)
        loaded_data.append(data)
    data_In.append(loaded_data)
else:
    # 如果没有.npy文件，添加空列表以保持结构
    data_In.append([])

# 从user_input读取input_data.csv文件
csv_path = "data/intermediate/user_input/input_data.csv"
try:
    # 处理UTF-8 BOM并读取CSV文件
    with open(csv_path, 'rb') as f:
        # 读取文件开头并检查BOM
        bom = f.read(3)
        if bom != b'\xef\xbb\xbf':
            # 如果没有BOM，重置文件指针
            f.seek(0)
        # 使用numpy从文件对象读取CSV，处理空值
        csv_data = np.genfromtxt(f, delimiter=',', dtype=float, filling_values=np.nan)
    
    # 提取第1列（所需template个数）存储到data_In的第2行
    template_counts = csv_data[:, 0].astype(int)  # 转换为整数
    data_In.append(template_counts.tolist())
    
    # 提取第2-5列（旋转角度）打包成一维数组存储到data_In的第3行
    angle_arrays = []
    for i in range(len(csv_data)):
        # 获取第2-5列，过滤掉NaN值
        angles = csv_data[i, 1:5]
        valid_angles = angles[~np.isnan(angles)].astype(float).tolist()
        angle_arrays.append(valid_angles)
    data_In.append(angle_arrays)
    
except Exception as e:
    print(f"读取CSV文件时出错: {e}")
    # 如果读取失败，添加空行以保持结构
    # 确保data_In有至少1个元素（模板数据）
    if len(data_In) == 0:
        data_In.append([])  # 第0行（模板数据）
    # 添加第1行（模板计数）和第2行（角度数组）
    while len(data_In) < 3:
        data_In.append([])

# 将每个零件的副本以及不同旋转角度下的实例(包含像素均值)添加到实际处理的数据列表actual_data_In中
actual_data_In = []
matrix_row = []
angle_row = []
t_mean_set = []
for i in range(len(data_In[0])):
    template_matrix = data_In[0][i]
    template_count = data_In[1][i]
    template_angle = data_In[2][i]
    
    # 将模板二值化（根据文档，模板应为二值矩阵）
    # 使用阈值将灰度图像转换为二值图像
    if template_matrix.max() > 1:  # 如果是灰度图像
        threshold = template_matrix.max() * 0.5  # 使用50%最大值为阈值
        template_binary = (template_matrix > threshold).astype(np.uint8)
    else:
        template_binary = template_matrix
    
    mean_value = np.mean(template_binary)
    for j in range(template_count):
        for k in range(len(template_angle)):  # 根据每个零件的旋转角度数量进行复制
            matrix_row.append(template_binary)
            angle_row.append(template_angle[k])
            t_mean_set.append(mean_value)
# 将矩阵行和角度行添加到actual_data_In中
actual_data_In.append(matrix_row)
actual_data_In.append(angle_row)
actual_data_In.append(t_mean_set)
# 对actual_data_In中的矩阵进行相应角度的旋转（步骤2：模板旋转）
for i in range(len(actual_data_In[0])):
    template_matrix = actual_data_In[0][i]
    angle = actual_data_In[1][i]
    
    # 计算旋转后的矩阵（仅支持90/180/270°旋转）
    # 注意：np.rot90是逆时针旋转，但我们需要确保旋转方向与DXF一致
    # 测试修改：尝试顺时针旋转
    if angle == 90:
        rotated_matrix = np.rot90(template_matrix, k=1)  # 逆时针90° (或顺时针270°)
    elif angle == 180:
        rotated_matrix = np.rot90(template_matrix, k=2)   # 逆时针180° (与顺时针180°相同)
    elif angle == 270:
        rotated_matrix = np.rot90(template_matrix, k=-1)   # 顺时针270° (或逆时针90°)
    elif angle == 0:
        rotated_matrix = template_matrix  # 0度不旋转
    else:
        # 非90倍数角度（如30°）需补充插值旋转，此处按0°处理并给出警告
        print(f"模板{i}：角度{angle}°不支持（非90倍数），按0°排布")
        rotated_matrix = template_matrix
    
    # 更新actual_data_In中的矩阵为旋转后的矩阵
    actual_data_In[0][i] = rotated_matrix
    
    # 记录旋转后模板尺寸（用于后续边界检查）
    rt_h, rt_w = rotated_matrix.shape
    # 可以存储尺寸信息，这里简单打印
    # print(f"模板{i}旋转后尺寸：{rt_h}×{rt_w}")

# 准备基板
base_width = SUBSTRATE_WIDTH_MM
base_height = SUBSTRATE_HEIGHT_MM
pixel_resolution = PIXEL_RESOLUTION
rows = int(base_height / pixel_resolution)
cols = int(base_width / pixel_resolution)
substrate_matrix = np.zeros((rows, cols), dtype=np.uint8) # 无边框


### 2.计算核心
# 反转基板（0和1互换） - 模板仅匹配未占用区域
substrate_inv = 1 - substrate_matrix
S = substrate_inv.astype(float)
P, Q = S.shape # 基板尺寸维度

# 互相关矩阵计算
c_set = []  # 存储每个模板的匹配度矩阵
for i in range(len(actual_data_In[0])):
    template_matrix = actual_data_In[0][i]
    # 使用compute_correlation_matrix函数计算相关性矩阵
    correlation_matrix = compute_correlation_matrix(template_matrix, substrate_inv)
    c_set.append(correlation_matrix)

print(f"计算完成，共 {len(c_set)} 个相关性矩阵")
print(f"第一个矩阵形状: {c_set[0].shape if len(c_set) > 0 else '无'}")


def get_best_match_positions(correlation_matrix, threshold=0.95):
    """
    从相关性矩阵中筛选符合阈值的最优匹配位置（降序）
    参数：
        correlation_matrix: NCC矩阵
        threshold: 匹配度阈值（仅保留≥阈值的位置）
    返回：
        sorted_positions: 按匹配度降序排列的位置列表 [(u, v, ncc_value), ...]
    """
    # 获取所有≥阈值的位置坐标和匹配度
    u_indices, v_indices = np.where(correlation_matrix >= threshold)
    valid_positions = [
        (u, v, correlation_matrix[u, v]) 
        for u, v in zip(u_indices, v_indices)
    ]
    # 按匹配度降序排序（优先选匹配度最高的位置）
    sorted_positions = sorted(valid_positions, key=lambda x: x[2], reverse=True)
    return sorted_positions

# 为每个模板计算最优匹配位置
best_positions_set = []
MATCH_THRESHOLD = 0.95  # 可根据业务调整
for c_matrix in c_set:
    best_pos = get_best_match_positions(c_matrix, MATCH_THRESHOLD)
    best_positions_set.append(best_pos)


def update_substrate(substrate_matrix, template_matrix, pos):
    """
    将模板放置到基板指定位置，并更新基板占用状态
    参数：
        substrate_matrix: 基板矩阵（0=未占用，1=已占用）
        template_matrix: 旋转后的模板矩阵
        pos: 放置位置 (u, v)
    返回：
        updated_substrate: 更新后的基板矩阵
        is_valid: 是否成功放置（无重叠）
    """
    u, v = pos
    m, n = template_matrix.shape
    P, Q = substrate_matrix.shape
    
    # 边界检查：模板不超出基板范围
    if u + m > P or v + n > Q:
        return substrate_matrix, False
    
    # 重叠检查：只检查模板中值为1的像素是否与基板中已占用的像素重叠
    patch = substrate_matrix[u:u+m, v:v+n]
    overlap_mask = (template_matrix == 1) & (patch == 1)
    if np.any(overlap_mask):
        return substrate_matrix, False
    
    # 更新基板：只将模板中值为1的像素设置为1
    updated_substrate = substrate_matrix.copy()
    substrate_patch = updated_substrate[u:u+m, v:v+n]
    mask = (template_matrix == 1)
    substrate_patch[mask] = 1
    updated_substrate[u:u+m, v:v+n] = substrate_patch
    return updated_substrate, True

# 简化主流程：使用arrange_single_template函数，按副本分组选择最优角度
print("\n=== 开始模板排布流程（按副本分组选择最优角度）===")
current_substrate = substrate_matrix.copy()
placed_templates = []  # 记录已放置的模板信息
unplaced_templates = []  # 记录未放置成功的模板

# 创建辅助列表同步记录实例信息
placement_records = []  # 每个元素: (template_idx, copy_idx, angle, center_x_mm, center_y_mm)

# 重建副本-角度映射关系
# data_In[1]是副本数，data_In[2]是角度数组
template_counts = data_In[1]  # 每个模板的副本数
angle_arrays = data_In[2]     # 每个模板的角度数组

print(f"模板数量: {len(template_counts)}")
print(f"总副本数: {sum(template_counts)}")

# 遍历每个模板
instance_idx = 0  # 在actual_data_In中的当前索引
for template_idx in range(len(template_counts)):
    copy_count = template_counts[template_idx]
    angles = angle_arrays[template_idx]
    angle_count = len(angles)
    
    print(f"\n模板{template_idx}: {copy_count}个副本, {angle_count}个角度({angles})")
    
    # 遍历该模板的每个副本
    for copy_idx in range(copy_count):
        # 该副本对应的所有角度实例的索引范围
        copy_start_idx = instance_idx + copy_idx * angle_count
        copy_end_idx = copy_start_idx + angle_count
        
        print(f"  副本{copy_idx}: 实例索引{copy_start_idx}-{copy_end_idx-1}")
        
        # 评估该副本所有角度实例的匹配度，选择最优角度
        best_placement = None
        best_success = False
        best_updated_substrate = None
        best_instance_idx = -1
        best_angle = 0
        best_physical_position = None
        
        # 遍历该副本的所有角度
        temp_unplaced = []  # 临时存储失败的角度索引
        for angle_idx in range(angle_count):
            instance_idx_current = copy_start_idx + angle_idx
            rotated_template = actual_data_In[0][instance_idx_current]
            target_angle = actual_data_In[1][instance_idx_current]
            
            # 调用单个模板排布函数，使用临时列表
            updated_substrate, placement_info, success = arrange_single_template(
                template_idx=instance_idx_current,
                template_matrix=rotated_template,
                target_angle=target_angle,
                substrate_matrix=current_substrate,
                unplaced_templates=temp_unplaced  # 使用临时列表
            )
            
            # 记录最佳匹配（按匹配度排序）
            # 修改：当匹配度相同时，优先选择较小的角度（0°优先于90°，90°优先于180°，等等）
            if success:
                if best_placement is None:
                    # 第一个成功的角度
                    best_placement = placement_info
                    best_success = success
                    best_updated_substrate = updated_substrate
                    best_instance_idx = instance_idx_current
                    best_angle = target_angle
                    best_physical_position = placement_info['physical_position']
                else:
                    # 比较匹配度
                    current_ncc = placement_info['ncc_value']
                    best_ncc = best_placement['ncc_value']
                    
                    if current_ncc > best_ncc:
                        # 当前匹配度更高，选择当前角度
                        best_placement = placement_info
                        best_success = success
                        best_updated_substrate = updated_substrate
                        best_instance_idx = instance_idx_current
                        best_angle = target_angle
                        best_physical_position = placement_info['physical_position']
                    elif abs(current_ncc - best_ncc) < 0.001:  # 匹配度基本相同
                        # 匹配度相同时，选择较小的角度（0° < 90° < 180° < 270°）
                        if target_angle < best_angle:
                            best_placement = placement_info
                            best_success = success
                            best_updated_substrate = updated_substrate
                            best_instance_idx = instance_idx_current
                            best_angle = target_angle
                            best_physical_position = placement_info['physical_position']
        
        # 如果找到最佳角度，使用它进行排布
        if best_success and best_placement is not None:
            current_substrate = best_updated_substrate
            placed_templates.append(best_placement)
            
            # 同步记录实例信息到辅助列表
            center_x_mm, center_y_mm = best_physical_position
            # 注意：不再进行角度转换，因为坐标系已经对齐
            # best_angle = (best_angle + 180) % 360  # 移除这个转换
            placement_records.append({
                'template_idx': template_idx,      # 零件编号（对应data_In中的编号）
                'copy_idx': copy_idx,              # 副本序号
                'angle': best_angle,               # 转角（直接使用，不转换）
                'center_x_mm': center_x_mm,        # 中心点X坐标
                'center_y_mm': center_y_mm,        # 中心点Y坐标
                'ncc_value': best_placement['ncc_value']  # 匹配度
            })
            
            print(f"    选择角度{best_angle}°，匹配度{best_placement['ncc_value']:.3f}")
            print(f"    记录实例信息: 零件{template_idx}, 副本{copy_idx}, 位置({center_x_mm:.1f}, {center_y_mm:.1f})")
        else:
            # 所有角度都失败，记录未放置
            print(f"    所有角度均无法放置")
            # 将第一个角度实例标记为未放置（代表该副本）
            first_instance_idx = copy_start_idx
            if first_instance_idx not in unplaced_templates:
                unplaced_templates.append(first_instance_idx)
    
    # 更新instance_idx到下一个模板的起始位置
    instance_idx += copy_count * angle_count

# 输出最终布局结果
print(f"\n=== 排布完成 ===")
print(f"成功放置 {len(placed_templates)} 个模板")
print(f"未放置 {len(unplaced_templates)} 个模板，索引：{unplaced_templates}")

# 打印所有已放置模板的摘要信息
if placed_templates:
    print(f"\n已放置模板摘要：")
    for i, item in enumerate(placed_templates[:5]):  # 只显示前5个
        print(f"  模板{item['template_idx']}: 位置{item['pixel_position']}, "
              f"匹配度{item['ncc_value']:.3f}, 角度{item['rotation_angle']}°")
    if len(placed_templates) > 5:
        print(f"  ... 还有{len(placed_templates)-5}个模板")
    
    # 详细显示第一个模板
    first_template = placed_templates[0]
    print(f"\n第一个模板详细信息：")
    print(f"  模板索引：{first_template['template_idx']}")
    print(f"  像素位置：{first_template['pixel_position']}")
    print(f"  物理位置（毫米）：{first_template['physical_position']}")
    print(f"  旋转角度：{first_template['rotation_angle']}°")
    print(f"  匹配度：{first_template['ncc_value']:.3f}")
    print(f"  模板尺寸：{first_template['template_size']}")

# 保存最终基板矩阵到文件，供可视化使用

import os
output_dir = "data/intermediate/final_substrate"
os.makedirs(output_dir, exist_ok=True)
substrate_output_path = os.path.join(output_dir, "final_substrate.npy")
np.save(substrate_output_path, current_substrate)
print(f"\n最终基板矩阵已保存到: {substrate_output_path}")
print(f"基板矩阵形状: {current_substrate.shape}")
print(f"已占用像素数量: {np.sum(current_substrate == 1)}")
print(f"空白像素数量: {np.sum(current_substrate == 0)}")


# 输出CSV格式数据到transformation_info文件夹
import os
import csv

# 创建transformation_info文件夹
transformation_dir = "data/intermediate/transformation_info"
os.makedirs(transformation_dir, exist_ok=True)
csv_path = os.path.join(transformation_dir, "placement_info.csv")

# 直接使用placement_records列表准备CSV数据
# 首先统计每个零件的副本个数
template_copy_counts = {}
for record in placement_records:
    template_idx = record['template_idx']
    copy_idx = record['copy_idx']
    if template_idx not in template_copy_counts:
        template_copy_counts[template_idx] = set()
    template_copy_counts[template_idx].add(copy_idx)

# 准备CSV数据
csv_data = []
csv_header = ["part_index", "copy_count", "copy_index", "angle", "center_x_mm", "center_y_mm"]

for record in placement_records:
    template_idx = record['template_idx']
    copy_idx = record['copy_idx']
    angle = record['angle']
    center_x_mm = record['center_x_mm']
    center_y_mm = record['center_y_mm']
    
    # 获取该零件的副本总数
    total_copies = len(template_copy_counts.get(template_idx, set()))
    
    csv_data.append([
        template_idx,          # 零件序号
        total_copies,          # 副本个数
        copy_idx,              # 副本序号
        angle,                 # 角度
        round(center_x_mm, 3), # 中心点X坐标
        round(center_y_mm, 3)  # 中心点Y坐标
    ])

# 按零件序号和副本序号排序
csv_data.sort(key=lambda x: (x[0], x[2]))

# 写入CSV文件
try:
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_data)
    print(f"\nCSV格式数据已保存到: {csv_path}")
    print(f"共记录 {len(csv_data)} 个副本的排布信息")
    print(f"数据来源: 直接同步记录的placement_records列表，无需map_instance_to_template映射")
except Exception as e:
    print(f"保存CSV文件时出错: {e}")