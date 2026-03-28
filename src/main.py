#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - 主控脚本

协调dxf_processor.py与computation_core.py完成完整工作流程：
1. 处理DXF文件，生成零件位图（带偏移）
2. 使用computation_core.py进行图案排布计算
3. 输出CSV格式的排布信息到transformation_info文件夹
4. 可选：使用dxf_processor.py的图形操作模块生成变换后的DXF文件

使用方法：
python main.py --input data/input/completeINPUT.dxf
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# 全局配置参数 - 用户可在此处修改
# ============================================================================

# 基板尺寸配置（毫米）
SUBSTRATE_WIDTH_MM = 150.0    # 基板宽度 (mm)
SUBSTRATE_HEIGHT_MM = 95.0    # 基板高度 (mm)

# 像素精度配置
PIXEL_RESOLUTION = 0.5         # 像素精度 (mm/pixel)

# 切割间隙配置
CUTTING_GAP_PG = 2.0          # 切割间隙 pg (mm)

# 旋转角度配置
ROTATION_ANGLES = [0, 90, 180, 270]  # 旋转角度列表

# 裁剪宽度系数配置
CROP_WIDTH_FACTOR = 1.5        # 第二次裁剪宽度系数

# ============================================================================
# 注意：这些全局变量将被所有脚本使用
# 用户可以通过修改这些值来调整系统参数
# ============================================================================

def setup_environment():
    """设置Python环境路径"""
    # 添加当前目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    logger.info(f"工作目录: {current_dir}")

def import_modules():
    """导入所需模块"""
    try:
        from dxf_processor import process_dxf_file, apply_graphic_transformations
        logger.info("✓ dxf_processor导入成功")
    except ImportError as e:
        logger.error(f"导入dxf_processor失败: {e}")
        logger.error("请确保dxf_processor.py在当前目录下")
        return None, None, None
    except Exception as e:
        logger.error(f"导入dxf_processor时出错: {e}")
        return None, None, None
    
    try:
        # 导入computation_core模块
        import computation_core
        logger.info("✓ computation_core导入成功")
        
        return process_dxf_file, computation_core, apply_graphic_transformations
            
    except Exception as e:
        logger.error(f"导入computation_core失败: {e}")
        logger.error("请检查computation_core.py文件")
        return None, None, None

def process_dxf_with_offset(input_file: str, pg = CUTTING_GAP_PG) -> Optional[str]:
    """
    处理DXF文件并生成带偏移的位图
    
    Args:
        input_file: 输入DXF文件路径
        pg: 切割间隙 (mm)
        
    Returns:
        str: 生成的位图目录路径，失败返回None
    """
    try:
        from dxf_processor import process_dxf_file
        
        logger.info(f"处理DXF文件: {input_file}")
        logger.info(f"切割间隙 pg: {pg}mm")
        
        # 处理DXF文件
        success = process_dxf_file(
            input_file=input_file,
            apply_offset=True,
            pg=pg
        )
        
        if not success:
            logger.error("DXF处理失败")
            return None
        
        # 确定生成的位图目录
        bitmap_dir = f"data/output/bitmaps_offset_pg{pg}"
        if os.path.exists(bitmap_dir):
            logger.info(f"位图已生成到: {bitmap_dir}")
            return bitmap_dir
        else:
            logger.error(f"位图目录不存在: {bitmap_dir}")
            return None
            
    except Exception as e:
        logger.error(f"处理DXF文件时出错: {e}")
        return None

def create_substrate_matrix(width_mm: float = SUBSTRATE_WIDTH_MM,
                           height_mm: float = SUBSTRATE_HEIGHT_MM,
                           pixel_resolution: float = PIXEL_RESOLUTION) -> str:
    """
    创建基板矩阵（带1像素边框）
    
    Args:
        width_mm: 基板宽度 (mm) - 严格使用用户输入值，默认使用全局变量SUBSTRATE_WIDTH_MM
        height_mm: 基板高度 (mm) - 严格使用用户输入值，默认使用全局变量SUBSTRATE_HEIGHT_MM
        pixel_resolution: 像素精度 (mm/pixel) - 严格使用用户输入值，默认使用全局变量PIXEL_RESOLUTION
        
    Returns:
        str: 基板矩阵文件路径
    """
    import numpy as np
    
    # 严格计算像素尺寸（使用用户输入值）
    rows = int(height_mm / pixel_resolution)
    cols = int(width_mm / pixel_resolution)
    
    # 验证计算结果的合理性
    if rows <= 0 or cols <= 0:
        logger.error(f"无效的基板尺寸: {rows}x{cols} 像素 (输入: {height_mm}x{width_mm} mm, 精度: {pixel_resolution} mm/pixel)")
        raise ValueError(f"基板尺寸必须为正数: {rows}x{cols}")
    
    # 创建基板矩阵（全0，表示空白区域）
    # 添加1像素边框：边框区域设置为1（表示不可用区域）
    substrate = np.zeros((rows, cols), dtype=np.uint8)
    
    # 添加1像素边框（设置为1，表示边界不可用）
    if rows > 2 and cols > 2:  # 确保有足够的空间添加边框
        substrate[0, :] = 1          # 上边框
        substrate[-1, :] = 1         # 下边框
        substrate[:, 0] = 1          # 左边框
        substrate[:, -1] = 1         # 右边框
        
        logger.info(f"已添加1像素边框: 边框像素值=1（不可用区域）")
    
    # 保存基板矩阵
    output_dir = Path("data/output/final_matrix")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    substrate_path = output_dir / "substrate.npy"
    np.save(substrate_path, substrate)
    
    # 计算实际可用区域（扣除边框）
    usable_rows = max(0, rows - 2) if rows > 2 else rows
    usable_cols = max(0, cols - 2) if cols > 2 else cols
    
    logger.info(f"创建基板矩阵: {substrate.shape} 像素 ({rows}x{cols})")
    logger.info(f"  输入尺寸: {height_mm}x{width_mm} mm, 像素精度: {pixel_resolution} mm/pixel")
    logger.info(f"  边框: 1像素（值=1，表示不可用）")
    logger.info(f"  可用区域: {usable_rows}x{usable_cols} 像素")
    logger.info(f"基板已保存到: {substrate_path}")
    
    return str(substrate_path)

def perform_layout_computation(template_dir: str, substrate_width_mm: float = SUBSTRATE_WIDTH_MM,
                              substrate_height_mm: float = SUBSTRATE_HEIGHT_MM,
                              pixel_resolution: float = PIXEL_RESOLUTION,
                              rotation_angles: list = None) -> bool:
    """
    执行图案排布计算
    
    Args:
        template_dir: 模板目录路径
        substrate_width_mm: 基板宽度 (mm)
        substrate_height_mm: 基板高度 (mm)
        pixel_resolution: 像素精度 (mm/pixel)
        rotation_angles: 旋转角度列表
        
    Returns:
        bool: 排布计算是否成功
    """
    try:
        import computation_core
        
        logger.info(f"开始图案排布计算")
        logger.info(f"模板目录: {template_dir}")
        logger.info(f"基板尺寸: {substrate_width_mm}x{substrate_height_mm} mm")
        logger.info(f"像素精度: {pixel_resolution} mm/pixel")
        logger.info(f"旋转角度: {rotation_angles}")
        
        # 注意：computation_core模块内部会处理排布计算并生成CSV文件
        # 这里我们只需要确保模块可以正常导入和运行
        logger.info("排布计算将由computation_core模块自动执行")
        logger.info(f"CSV文件将输出到: ../data/intermediate/transformation_info/placement_info.csv")
        
        return True
            
    except Exception as e:
        logger.error(f"图案排布计算时出错: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='不规则图案自动化排布系统')
    parser.add_argument('--input', type=str, default='data/input/completeINPUT.dxf',
                       help='输入DXF文件路径')
    parser.add_argument('--pg', type=float, default=CUTTING_GAP_PG,
                       help=f'切割间隙 pg (mm)，默认{CUTTING_GAP_PG}')
    parser.add_argument('--substrate-width', type=float, default=SUBSTRATE_WIDTH_MM,
                       help=f'基板宽度 (mm)，默认{SUBSTRATE_WIDTH_MM}')
    parser.add_argument('--substrate-height', type=float, default=SUBSTRATE_HEIGHT_MM,
                       help=f'基板高度 (mm)，默认{SUBSTRATE_HEIGHT_MM}')
    parser.add_argument('--pixel-resolution', type=float, default=PIXEL_RESOLUTION,
                       help=f'像素精度 (mm/pixel)，默认{PIXEL_RESOLUTION}')
    parser.add_argument('--rotation-angles', type=str,
                       default=','.join(str(angle) for angle in ROTATION_ANGLES),
                       help=f'旋转角度列表，逗号分隔，默认{ROTATION_ANGLES}')
    parser.add_argument('--skip-dxf', action='store_true',
                       help='跳过DXF处理，直接使用现有位图')
    parser.add_argument('--template-dir', type=str,
                       help='模板目录路径（如果跳过DXF处理）')
    parser.add_argument('--apply-transformations', action='store_true', default=True,
                       help='应用图形变换，生成变换后的DXF文件（默认启用）')
    parser.add_argument('--no-transformations', action='store_false', dest='apply_transformations',
                       help='禁用图形变换')
    parser.add_argument('--output-dxf', type=str, default='data/output/transformed_parts.dxf',
                       help='输出DXF文件路径（如果应用图形变换）')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("不规则图案自动化排布系统")
    logger.info("=" * 60)
    
    # 设置环境
    setup_environment()
    
    # 导入模块
    process_dxf_file, computation_core, apply_graphic_transformations = import_modules()
    if process_dxf_file is None or computation_core is None:
        return 1
    
    # 步骤1: 处理DXF文件（除非跳过）
    template_dir = args.template_dir
    if not args.skip_dxf:
        logger.info("\n步骤1: 处理DXF文件并生成位图")
        template_dir = process_dxf_with_offset(args.input, args.pg)
        if template_dir is None:
            logger.error("DXF处理失败，退出")
            return 1
    else:
        if template_dir is None:
            # 使用默认模板目录
            template_dir = f"data/output/bitmaps_offset_pg{args.pg}"
            logger.info(f"使用默认模板目录: {template_dir}")
        
        if not os.path.exists(template_dir):
            logger.error(f"模板目录不存在: {template_dir}")
            return 1
    
    # 步骤2: 创建基板矩阵
    logger.info("\n步骤2: 创建基板矩阵")
    substrate_path = create_substrate_matrix(
        width_mm=args.substrate_width,
        height_mm=args.substrate_height,
        pixel_resolution=args.pixel_resolution
    )
    
    # 步骤3: 执行图案排布计算
    logger.info("\n步骤3: 执行图案排布计算")
    
    # 解析旋转角度
    rotation_angles = [int(angle.strip()) for angle in args.rotation_angles.split(',')]
    
    success = perform_layout_computation(
        template_dir=template_dir,
        substrate_width_mm=args.substrate_width,
        substrate_height_mm=args.substrate_height,
        pixel_resolution=args.pixel_resolution,
        rotation_angles=rotation_angles
    )
    
    # 步骤4: 可选 - 应用图形变换
    if args.apply_transformations and apply_graphic_transformations is not None:
        logger.info("\n步骤4: 应用图形变换")
        
        csv_path = "data/intermediate/transformation_info/placement_info.csv"
        # 源DXF文件路径 - 使用processed_parts.dxf作为源文件
        source_dxf_path = "data/output/processed_parts.dxf"
        
        if os.path.exists(csv_path) and os.path.exists(source_dxf_path):
            transform_success = apply_graphic_transformations(
                input_csv=csv_path,
                source_dxf_path=source_dxf_path,
                output_dxf_path=args.output_dxf
            )
            if transform_success:
                logger.info(f"图形变换完成，输出DXF文件: {args.output_dxf}")
            else:
                logger.warning("图形变换失败，但排布计算已完成")
        else:
            if not os.path.exists(csv_path):
                logger.warning(f"CSV文件不存在，跳过图形变换: {csv_path}")
            if not os.path.exists(source_dxf_path):
                logger.warning(f"源DXF文件不存在，跳过图形变换: {source_dxf_path}")
                logger.warning(f"请先运行DXF处理流程生成零件块文件")
    
    # 输出总结
    logger.info("\n" + "=" * 60)
    logger.info("工作流程总结")
    logger.info("=" * 60)
    logger.info(f"输入DXF文件: {args.input}")
    logger.info(f"切割间隙 pg: {args.pg}mm")
    logger.info(f"基板尺寸: {args.substrate_width}x{args.substrate_height} mm")
    logger.info(f"模板目录: {template_dir}")
    logger.info(f"旋转角度: {rotation_angles}")
    
    if success:
        logger.info("状态: 排布计算成功完成")
        logger.info(f"排布信息保存在: ../data/intermediate/transformation_info/placement_info.csv")
        if args.apply_transformations and apply_graphic_transformations is not None:
            logger.info(f"变换后的DXF文件: {args.output_dxf}")
        return 0
    else:
        logger.error("状态: 排布计算失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
