#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gui_app.py — Arrangeit 图形界面

基于 PySide6，提供 3 步向导式操作：
  1. 选择 DXF、设置基板参数、处理文件
  2. 浏览零件卡片、设置副本数与旋转角度
  3. 运行排样计算、查看结果、导出
"""

import sys, os, json, csv, subprocess, shutil, tempfile
from pathlib import Path
import numpy as np

# ── PySide6 imports ───────────────────────────────────────────
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QLineEdit, QSpinBox, QCheckBox,
    QScrollArea, QFrame, QGridLayout, QGroupBox, QProgressBar,
    QMessageBox, QStackedWidget, QSplitter, QDoubleSpinBox,
    QSizePolicy, QButtonGroup, QRadioButton, QDialog, QVBoxLayout as QVBL,
)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QTimer
from PySide6.QtGui import QPixmap, QFont, QPalette, QColor

# ── 项目路径设置 ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(SRC_DIR))

# ── 全局配置默认值 ────────────────────────────────────────────
DEFAULT_SUBSTRATE_W = 150.0
DEFAULT_SUBSTRATE_H = 95.0
DEFAULT_PIXEL_RES   = 0.5
DEFAULT_CUTTING_GAP = 2.0
ROTATION_OPTIONS    = [0, 90, 180, 270]

# ╔══════════════════════════════════════════════════════════════╗
# ║                    DATA  CLEANUP                             ║
# ╚══════════════════════════════════════════════════════════════╝

def _clean_previous_outputs():
    """清理上一次运行产生的所有中间/输出数据，防止新旧数据混淆"""
    dirs_to_clean = [
        DATA_DIR / "intermediate" / "template_matrix",
        DATA_DIR / "intermediate" / "final_substrate",
        DATA_DIR / "intermediate" / "transformation_info",
        DATA_DIR / "intermediate" / "user_input",
    ]
    files_to_clean = [
        DATA_DIR / "output" / "parts_all.png",
        DATA_DIR / "output" / "parts_outer_only.png",
        DATA_DIR / "output" / "processed_parts.dxf",
        DATA_DIR / "output" / "transformed_parts.dxf",
        DATA_DIR / "output" / "substrate_visualization.png",
        DATA_DIR / "output" / "substrate_binary.png",
        DATA_DIR / "output" / "substrate_physical.png",
        DATA_DIR / "intermediate" / "run_config.json",
    ]
    # 清理位图目录（匹配 pg 参数的所有变体）
    bitmap_parent = DATA_DIR / "output"
    if bitmap_parent.exists():
        for child in bitmap_parent.iterdir():
            if child.is_dir() and child.name.startswith("bitmaps_offset_pg"):
                dirs_to_clean.append(child)

    for d in dirs_to_clean:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    for f in files_to_clean:
        if f.exists():
            try:
                f.unlink()
            except OSError:
                pass


# ╔══════════════════════════════════════════════════════════════╗
# ║                     WORKER  THREADS                          ║
# ╚══════════════════════════════════════════════════════════════╝

class DxfProcessWorker(QThread):
    """后台处理 DXF 文件（调用 dxf_processor）"""
    progress = Signal(str)          # 日志消息
    finished_ok = Signal(bool, str) # 成功与否, 位图目录路径
    part_count = Signal(int)        # 识别到的零件数

    def __init__(self, input_file: str, pg: float):
        super().__init__()
        self.input_file = input_file
        self.pg = pg

    def run(self):
        try:
            # ── 清理上一次运行残留的旧数据 ──
            self.progress.emit("正在清理旧数据...")
            _clean_previous_outputs()

            self.progress.emit("正在导入 dxf_processor 模块...")
            from dxf_processor import process_dxf_file

            self.progress.emit(f"正在处理 DXF: {self.input_file}  (pg={self.pg}mm)...")
            success = process_dxf_file(
                input_file=self.input_file,
                apply_offset=True,
                pg=self.pg,
            )

            if not success:
                self.finished_ok.emit(False, "DXF 处理返回失败")
                return

            # 确定生成的位图目录
            bitmap_dir = DATA_DIR / "output" / f"bitmaps_offset_pg{self.pg}"
            if not bitmap_dir.exists():
                self.finished_ok.emit(False, f"位图目录不存在: {bitmap_dir}")
                return

            # 统计零件数量
            pngs = sorted(bitmap_dir.glob("part_*.png"))
            self.part_count.emit(len(pngs))
            self.progress.emit(f"✓ 处理完成，识别到 {len(pngs)} 个零件")
            self.finished_ok.emit(True, str(bitmap_dir))

        except Exception as e:
            self.progress.emit(f"✗ 错误: {e}")
            self.finished_ok.emit(False, str(e))


class ComputeWorker(QThread):
    """后台运行排样计算"""
    progress = Signal(str)
    finished_ok = Signal(bool, str)  # 成功, csv_path

    def __init__(self, substrate_w: float, substrate_h: float,
                 pixel_res: float, rotation_angles: list,
                 pg: float):
        super().__init__()
        self.substrate_w = substrate_w
        self.substrate_h = substrate_h
        self.pixel_res = pixel_res
        self.rotation_angles = rotation_angles
        self.pg = pg

    def run(self):
        try:
            # 写入运行配置供 computation_core 读取
            config = {
                "SUBSTRATE_WIDTH_MM":  self.substrate_w,
                "SUBSTRATE_HEIGHT_MM": self.substrate_h,
                "PIXEL_RESOLUTION":    self.pixel_res,
                "CUTTING_GAP_PG":      self.pg,
                "ROTATION_ANGLES":     self.rotation_angles,
            }
            config_path = DATA_DIR / "intermediate" / "run_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

            self.progress.emit("正在运行排样计算...")
            comp_script = SRC_DIR / "computation_core.py"
            proc = subprocess.run(
                [sys.executable, str(comp_script)],
                cwd=str(PROJECT_ROOT),
                capture_output=True, text=True, timeout=600,
            )

            if proc.returncode != 0:
                self.progress.emit(f"计算错误:\n{proc.stderr[-800:]}")
                self.finished_ok.emit(False, proc.stderr)
                return

            # 输出最后几行日志
            for line in proc.stdout.strip().splitlines()[-15:]:
                self.progress.emit(line)

            csv_path = DATA_DIR / "intermediate" / "transformation_info" / "placement_info.csv"
            if csv_path.exists():
                self.progress.emit(f"✓ 排样完成，结果保存至: {csv_path}")
                # 自动生成基板可视化图
                self.progress.emit("正在生成基板可视化...")
                viz_script = SRC_DIR / "visualize_substrate.py"
                subprocess.run(
                    [sys.executable, str(viz_script)],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True, text=True, timeout=60,
                )
                self.finished_ok.emit(True, str(csv_path))
            else:
                self.finished_ok.emit(False, "未找到 placement_info.csv 输出文件")

        except subprocess.TimeoutExpired:
            self.finished_ok.emit(False, "计算超时（>10分钟）")
        except Exception as e:
            self.finished_ok.emit(False, str(e))


# ╔══════════════════════════════════════════════════════════════╗
# ║                  PART  CARD  WIDGET                          ║
# ╚══════════════════════════════════════════════════════════════╝

class PartCard(QFrame):
    """单个零件的配置卡片：缩略图 + 副本数 + 旋转角度"""

    def __init__(self, part_index: int, image_path: str, parent=None):
        super().__init__(parent)
        self.part_index = part_index
        self.setObjectName("partCard")
        self.setStyleSheet("""
            #partCard {
                background: #ffffff;
                border: 1px solid #c0c0c0;
                border-radius: 8px;
                padding: 6px;
            }
            #partCard:hover { border: 1px solid #4a90d9; background: #f8faff; }
        """)
        self.setFixedWidth(180)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)

        # ── 标题 ──
        title = QLabel(f"零件 #{part_index}")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        title.setStyleSheet("color: #1a1a1a;")
        layout.addWidget(title)

        # ── 缩略图 ──
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setFixedSize(140, 100)
        self.img_label.setStyleSheet("background: #fafafa; border: 1px solid #d5d5d5;")
        self.img_label.setScaledContents(False)
        layout.addWidget(self.img_label, alignment=Qt.AlignCenter)

        # 加载缩略图
        if os.path.exists(image_path):
            pix = QPixmap(image_path)
            if not pix.isNull():
                pix = pix.scaled(136, 96, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.img_label.setPixmap(pix)
        else:
            self.img_label.setText("无图像")

        # ── 分离线 ──
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background: #d0d0d0; max-height: 1px;")
        layout.addWidget(sep)

        # ── 副本数 ──
        copy_layout = QHBoxLayout()
        copy_label = QLabel("副本:")
        copy_label.setStyleSheet("color: #1a1a1a;")
        copy_layout.addWidget(copy_label)
        self.copy_spin = QSpinBox()
        self.copy_spin.setRange(0, 99)
        self.copy_spin.setValue(1)
        self.copy_spin.setFixedWidth(60)
        self.copy_spin.valueChanged.connect(self._on_copy_changed)
        copy_layout.addWidget(self.copy_spin)
        copy_layout.addStretch()
        layout.addLayout(copy_layout)

        # ── 旋转角度 ──
        angle_label = QLabel("旋转角度:")
        angle_label.setStyleSheet("color: #1a1a1a; font-size: 9pt;")
        layout.addWidget(angle_label)

        self.angle_checks = {}
        angle_grid = QGridLayout()
        for i, deg in enumerate(ROTATION_OPTIONS):
            cb = QCheckBox(f"{deg}°")
            cb.setChecked(True)  # 默认全选
            self.angle_checks[deg] = cb
            angle_grid.addWidget(cb, i // 2, i % 2)
        layout.addLayout(angle_grid)

        layout.addStretch()

    def _on_copy_changed(self, val):
        """副本数为 0 时禁用角度选择"""
        enabled = val > 0
        for cb in self.angle_checks.values():
            cb.setEnabled(enabled)
        if not enabled:
            for cb in self.angle_checks.values():
                cb.setChecked(False)

    def get_copy_count(self) -> int:
        return self.copy_spin.value()

    def get_selected_angles(self) -> list:
        if self.copy_spin.value() == 0:
            return []
        return sorted([deg for deg, cb in self.angle_checks.items() if cb.isChecked()])


# ╔══════════════════════════════════════════════════════════════╗
# ║                         PAGES                                ║
# ╚══════════════════════════════════════════════════════════════╝

class SetupPage(QWidget):
    """第 1 页：选择 DXF + 设置基板参数 + 处理"""

    dxf_processed = Signal(str)   # 位图目录路径
    part_count_signal = Signal(int)

    def __init__(self):
        super().__init__()
        self.bitmap_dir = ""
        self.part_count = 0
        self._worker = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(15)

        # 标题
        title = QLabel("📋 Step 1 — 输入设置")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title.setStyleSheet("color: #1a1a1a;")
        layout.addWidget(title)

        # ── DXF 文件选择 ──
        file_group = QGroupBox("DXF 文件")
        file_ly = QHBoxLayout(file_group)
        self.dxf_path_edit = QLineEdit()
        self.dxf_path_edit.setPlaceholderText("选择 .dxf 文件...")
        self.dxf_path_edit.setReadOnly(True)
        file_ly.addWidget(self.dxf_path_edit)
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._browse_dxf)
        file_ly.addWidget(browse_btn)
        layout.addWidget(file_group)

        # ── 基板参数 ──
        param_group = QGroupBox("基板参数")
        param_grid = QGridLayout(param_group)
        param_grid.setSpacing(10)

        param_grid.addWidget(QLabel("宽度 (mm):"), 0, 0)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(10, 5000)
        self.width_spin.setValue(DEFAULT_SUBSTRATE_W)
        self.width_spin.setDecimals(1)
        param_grid.addWidget(self.width_spin, 0, 1)

        param_grid.addWidget(QLabel("高度 (mm):"), 1, 0)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(10, 5000)
        self.height_spin.setValue(DEFAULT_SUBSTRATE_H)
        self.height_spin.setDecimals(1)
        param_grid.addWidget(self.height_spin, 1, 1)

        param_grid.addWidget(QLabel("像素精度 (mm/px):"), 2, 0)
        self.res_spin = QDoubleSpinBox()
        self.res_spin.setRange(0.1, 5.0)
        self.res_spin.setValue(DEFAULT_PIXEL_RES)
        self.res_spin.setDecimals(2)
        self.res_spin.setSingleStep(0.05)
        param_grid.addWidget(self.res_spin, 2, 1)

        param_grid.addWidget(QLabel("切割间隙 (mm):"), 3, 0)
        self.gap_spin = QDoubleSpinBox()
        self.gap_spin.setRange(0.0, 50.0)
        self.gap_spin.setValue(DEFAULT_CUTTING_GAP)
        self.gap_spin.setDecimals(1)
        param_grid.addWidget(self.gap_spin, 3, 1)

        layout.addWidget(param_group)

        # ── 显示计算出的像素尺寸 ──
        self.pixel_info = QLabel()
        self.pixel_info.setStyleSheet("color: #555; font-size: 9pt;")
        layout.addWidget(self.pixel_info)
        self._update_pixel_info()

        self.width_spin.valueChanged.connect(lambda: self._update_pixel_info())
        self.height_spin.valueChanged.connect(lambda: self._update_pixel_info())
        self.res_spin.valueChanged.connect(lambda: self._update_pixel_info())

        # ── 处理按钮 ──
        self.process_btn = QPushButton("⚙ 处理 DXF 文件")
        self.process_btn.setFixedHeight(40)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background: #4a90d9; color: white; font-size: 12pt;
                border-radius: 6px; font-weight: bold;
            }
            QPushButton:hover { background: #357abd; }
            QPushButton:disabled { background: #cccccc; }
        """)
        self.process_btn.clicked.connect(self._process_dxf)
        layout.addWidget(self.process_btn)

        # ── 进度条 ──
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # 不确定模式
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # ── 日志区 ──
        self.log_area = QLabel()
        self.log_area.setWordWrap(True)
        self.log_area.setStyleSheet("color: #2c3e50; font-size: 9pt; background: #fafafa; padding: 4px;")
        layout.addWidget(self.log_area)

        layout.addStretch()

    def _update_pixel_info(self):
        w = self.width_spin.value()
        h = self.height_spin.value()
        r = self.res_spin.value()
        cols = int(w / r)
        rows = int(h / r)
        self.pixel_info.setText(f"→ 基板像素: {cols}×{rows} px")

    def _browse_dxf(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 DXF 文件", str(DATA_DIR / "input"),
            "DXF Files (*.dxf);;All Files (*.*)")
        if path:
            self.dxf_path_edit.setText(path)

    def _process_dxf(self):
        dxf_path = self.dxf_path_edit.text().strip()
        if not dxf_path or not os.path.exists(dxf_path):
            msg_warn(self, "路径错误", "请先选择一个有效的 DXF 文件。")
            return

        # 禁用按钮，显示进度
        self.process_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.log_area.setText("")

        # 直接使用用户选择的 DXF 文件路径（避免复制导致的权限问题）
        self._worker = DxfProcessWorker(dxf_path, self.gap_spin.value())
        self._worker.progress.connect(self._on_log)
        self._worker.finished_ok.connect(self._on_processed)
        self._worker.part_count.connect(self._on_part_count)
        self._worker.start()

    def _on_log(self, msg: str):
        self.log_area.setText(self.log_area.text() + msg + "\n")

    def _on_part_count(self, count: int):
        self.part_count = count

    def _on_processed(self, success: bool, info: str):
        self.process_btn.setEnabled(True)
        self.progress.setVisible(False)
        if success:
            self.bitmap_dir = info
            self.log_area.setText(self.log_area.text() + f"\n✅ 处理完成！共 {self.part_count} 个零件\n")
            self.dxf_processed.emit(info)
            self.part_count_signal.emit(self.part_count)
        else:
            self.log_area.setText(self.log_area.text() + f"\n❌ 处理失败: {info}\n")
            msg_err(self, "处理失败", info)

    def get_config(self):
        return {
            "substrate_w": self.width_spin.value(),
            "substrate_h": self.height_spin.value(),
            "pixel_res":   self.res_spin.value(),
            "cutting_gap": self.gap_spin.value(),
            "bitmap_dir":  self.bitmap_dir,
        }


class ConfigPage(QWidget):
    """第 2 页：零件卡片网格 + 总体预览"""

    config_saved = Signal(str)  # CSV 路径

    def __init__(self):
        super().__init__()
        self.bitmap_dir = ""
        self.cards: list[PartCard] = []
        self._parts_all_path = ""

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 20, 30, 20)

        # 标题
        title = QLabel("📋 Step 2 — 配置零件")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title.setStyleSheet("color: #1a1a1a;")
        main_layout.addWidget(title)

        # 说明文字
        hint = QLabel("副本数 = 该零件需要几个  |  角度 = 每个副本允许以哪些方向放置（程序会为每个副本自动选择最佳角度）")
        hint.setStyleSheet("color: #555; font-size: 9pt; padding: 2px 0 6px 0;")
        main_layout.addWidget(hint)

        # ── 顶部：预览图 + 全选工具条 ──
        top_bar = QHBoxLayout()

        # 零件总览预览图（可点击放大）
        self.preview_btn = QPushButton()
        self.preview_btn.setFixedSize(240, 140)
        self.preview_btn.setStyleSheet("background: #fafafa; border: 1px solid #b0b0b0; border-radius: 4px; color: #555;")
        self.preview_btn.setToolTip("点击查看大图")
        self.preview_btn.clicked.connect(self._show_full_preview)
        top_bar.addWidget(self.preview_btn)

        top_bar.addSpacing(20)

        # 批量操作面板
        quick_box = QGroupBox("批量操作")
        quick_box.setFixedWidth(320)
        quick_ly = QVBoxLayout(quick_box)
        quick_ly.setSpacing(6)

        # ── 批量副本数（勾选框 + 输入框 + 应用按钮）──
        copy_batch_row = QHBoxLayout()

        self.copy_enable_cb = QCheckBox("批量设置副本数")
        self.copy_enable_cb.toggled.connect(self._on_copy_batch_toggled)
        copy_batch_row.addWidget(self.copy_enable_cb)

        copy_batch_row.addWidget(QLabel("数量:"))

        self.copy_batch_spin = QSpinBox()
        self.copy_batch_spin.setRange(0, 99)
        self.copy_batch_spin.setValue(1)
        self.copy_batch_spin.setFixedWidth(60)
        self.copy_batch_spin.setEnabled(False)
        copy_batch_row.addWidget(self.copy_batch_spin)

        self.copy_apply_btn = QPushButton("应用到全部")
        self.copy_apply_btn.setEnabled(False)
        self.copy_apply_btn.clicked.connect(self._apply_batch_copies)
        copy_batch_row.addWidget(self.copy_apply_btn)

        copy_batch_row.addStretch()
        quick_ly.addLayout(copy_batch_row)

        # ── 批量角度（4 个勾选框 + 应用按钮）──
        angle_batch_label = QLabel("批量设置旋转角度:")
        angle_batch_label.setStyleSheet("color: #1a1a1a;")
        quick_ly.addWidget(angle_batch_label)

        angle_batch_row = QHBoxLayout()
        self.angle_batch_checks = {}
        for deg in ROTATION_OPTIONS:
            cb = QCheckBox(f"{deg}°")
            self.angle_batch_checks[deg] = cb
            angle_batch_row.addWidget(cb)

        self.angle_apply_btn = QPushButton("应用到全部")
        self.angle_apply_btn.clicked.connect(self._apply_batch_angles)
        angle_batch_row.addWidget(self.angle_apply_btn)
        angle_batch_row.addStretch()
        quick_ly.addLayout(angle_batch_row)

        quick_ly.addStretch()
        top_bar.addWidget(quick_box)
        top_bar.addStretch()

        main_layout.addLayout(top_bar)

        # ── 零件卡片滚动区 ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        self.card_container = QWidget()
        self.card_grid = QGridLayout(self.card_container)
        self.card_grid.setSpacing(12)
        self.card_grid.setContentsMargins(5, 5, 5, 5)
        scroll.setWidget(self.card_container)

        main_layout.addWidget(scroll, stretch=1)

        # ── 底部按钮 ──
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.run_btn = QPushButton("▶ 运行排样计算")
        self.run_btn.setFixedHeight(40)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background: #27ae60; color: white; font-size: 12pt;
                border-radius: 6px; font-weight: bold; padding: 0 30px;
            }
            QPushButton:hover { background: #219a52; }
            QPushButton:disabled { background: #cccccc; }
        """)
        self.run_btn.clicked.connect(self._save_and_go)
        btn_row.addWidget(self.run_btn)
        btn_row.addStretch()
        main_layout.addLayout(btn_row)

    def load_parts(self, bitmap_dir: str):
        """根据位图目录加载零件卡片；传入空字符串则仅清理"""
        self.bitmap_dir = bitmap_dir

        # 清除旧卡片
        for card in self.cards:
            card.setParent(None)
        self.cards.clear()
        while self.card_grid.count():
            item = self.card_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # 空路径 = 仅清理，不加载
        if not bitmap_dir:
            self.preview_btn.setIcon(QPixmap())
            self.preview_btn.setText("")
            self._parts_all_path = ""
            return

        # 查找零件预览总图
        parts_all = DATA_DIR / "output" / "parts_all.png"
        if parts_all.exists():
            self._parts_all_path = str(parts_all)
            pix = QPixmap(str(parts_all))
            pix = pix.scaled(236, 136, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview_btn.setIcon(pix)
            self.preview_btn.setIconSize(pix.size())
        else:
            self.preview_btn.setText("(预览图未生成)")

        # 加载每个零件的位图
        png_dir = Path(bitmap_dir)
        pngs = sorted(png_dir.glob("part_*.png"))

        if not pngs:
            msg_warn(self, "无零件", f"在 {bitmap_dir} 中未找到零件位图。")
            return

        cols = max(1, self.width() // 220)  # 每行卡片数

        for i, png_path in enumerate(pngs):
            card = PartCard(i, str(png_path))
            self.cards.append(card)
            row, col = divmod(i, cols)
            self.card_grid.addWidget(card, row, col)

        # 加载已有配置（如果存在）
        self._load_existing_config()

    def _load_existing_config(self):
        """尝试加载已有的 input_data.csv 来恢复配置"""
        csv_path = DATA_DIR / "intermediate" / "user_input" / "input_data.csv"
        if not csv_path.exists():
            return
        try:
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                rows = list(reader)
            if not rows:
                return
            for i, row in enumerate(rows):
                if i >= len(self.cards):
                    break
                if len(row) < 1:
                    continue
                try:
                    count = int(float(row[0]))
                except ValueError:
                    count = 0
                self.cards[i].copy_spin.setValue(count)

                # 解析角度（第2列及以后 = 允许的角度值）
                allowed = set()
                for val in row[1:]:
                    try:
                        allowed.add(int(float(val)))
                    except (ValueError, TypeError):
                        pass
                for deg, cb in self.cards[i].angle_checks.items():
                    cb.setChecked(deg in allowed)
        except Exception:
            pass  # 配置读取失败，使用默认值

    def _set_all_copies(self, val: int):
        for card in self.cards:
            card.copy_spin.setValue(val)

    # ── 批量操作：副本数 ──

    def _on_copy_batch_toggled(self, checked: bool):
        """勾选启用时高亮输入框，取消时置灰"""
        self.copy_batch_spin.setEnabled(checked)
        self.copy_apply_btn.setEnabled(checked)
        if checked:
            self.copy_batch_spin.setStyleSheet(
                "border: 2px solid #4a90d9; border-radius: 4px; padding: 4px 8px;"
                "background: #ffffff; color: #1a1a1a;"
            )
        else:
            self.copy_batch_spin.setStyleSheet("")

    def _apply_batch_copies(self):
        """将批量副本数应用到所有零件卡片"""
        val = self.copy_batch_spin.value()
        self._set_all_copies(val)

    # ── 批量操作：角度 ──

    def _apply_batch_angles(self):
        """将批量角度勾选状态应用到所有零件卡片"""
        target = {deg: cb.isChecked() for deg, cb in self.angle_batch_checks.items()}
        for card in self.cards:
            if card.copy_spin.value() > 0:
                for deg, cb in card.angle_checks.items():
                    cb.setChecked(target.get(deg, False))

    def _set_all_angles(self, checked: bool):
        for card in self.cards:
            if card.copy_spin.value() > 0:
                for cb in card.angle_checks.values():
                    cb.setChecked(checked)

    def _show_full_preview(self):
        if not self._parts_all_path or not os.path.exists(self._parts_all_path):
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("零件总览")
        dlg.setMinimumSize(600, 400)
        layout = QVBL(dlg)
        img = QLabel()
        pix = QPixmap(self._parts_all_path)
        pix = pix.scaled(900, 700, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        img.setPixmap(pix)
        img.setAlignment(Qt.AlignCenter)
        layout.addWidget(img)
        dlg.exec()

    def _save_and_go(self):
        """保存配置到 input_data.csv 并发出信号"""
        csv_path = DATA_DIR / "intermediate" / "user_input" / "input_data.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for card in self.cards:
            count = card.get_copy_count()
            angles = card.get_selected_angles()
            row = [count] + angles
            rows.append(row)

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        msg_info(self, "已保存", f"配置已保存至:\n{csv_path}\n共 {len(rows)} 个零件。")
        self.config_saved.emit(str(csv_path))

    def get_bitmap_dir(self) -> str:
        return self.bitmap_dir


class ResultPage(QWidget):
    """第 3 页：显示排样结果"""

    restart_requested = Signal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(15)

        title = QLabel("📋 Step 3 — 排样结果")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title.setStyleSheet("color: #1a1a1a;")
        layout.addWidget(title)

        # ── 可视化图 ──
        self.viz_label = QLabel()
        self.viz_label.setAlignment(Qt.AlignCenter)
        self.viz_label.setMinimumSize(400, 250)
        self.viz_label.setStyleSheet("background: #ffffff; border: 1px solid #b0b0b0;")
        layout.addWidget(self.viz_label, stretch=1)

        # ── 统计 ──
        self.stats_label = QLabel()
        self.stats_label.setFont(QFont("Microsoft YaHei", 11))
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("color: #1a1a1a;")
        layout.addWidget(self.stats_label)

        # ── 按钮 ──
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.export_dxf_btn = QPushButton("📁 导出变换后的 DXF")
        self.export_dxf_btn.setFixedHeight(36)
        self.export_dxf_btn.clicked.connect(self._export_dxf)
        btn_row.addWidget(self.export_dxf_btn)

        self.export_img_btn = QPushButton("🖼 导出基板图像")
        self.export_img_btn.setFixedHeight(36)
        self.export_img_btn.clicked.connect(self._export_image)
        btn_row.addWidget(self.export_img_btn)

        restart_btn = QPushButton("🔄 重新开始")
        restart_btn.setFixedHeight(36)
        restart_btn.clicked.connect(self.restart_requested.emit)
        btn_row.addWidget(restart_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

    def show_results(self, csv_path: str):
        """加载并显示结果"""
        # 加载基板可视化
        viz_paths = [
            DATA_DIR / "output" / "substrate_visualization.png",
            DATA_DIR / "output" / "substrate_binary.png",
        ]
        shown = False
        for vp in viz_paths:
            if vp.exists():
                pix = QPixmap(str(vp))
                pix = pix.scaled(700, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.viz_label.setPixmap(pix)
                shown = True
                break
        if not shown:
            self.viz_label.setText("(基板可视化图未找到)")

        # 统计信息
        stats_lines = []
        try:
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                rows = list(reader)

            if rows:
                part_ids = set(int(r[0]) for r in rows if len(r) > 0)
                stats_lines.append(f"✅ 成功放置: {len(rows)} 个副本")
                stats_lines.append(f"   涉及零件: {len(part_ids)} 种")
                stats_lines.append(f"   零件编号: {sorted(part_ids)}")
            else:
                stats_lines.append("⚠ 没有放置任何零件。")
        except Exception as e:
            stats_lines.append(f"读取结果出错: {e}")

        # 基板占用率
        final_sub = DATA_DIR / "intermediate" / "final_substrate" / "final_substrate.npy"
        if final_sub.exists():
            mat = np.load(final_sub)
            total = mat.size
            occupied = int(np.sum(mat == 1))
            pct = occupied / total * 100 if total > 0 else 0
            stats_lines.append(f"\n📊 基板利用率: {pct:.1f}% ({occupied}/{total} px)")

        self.stats_label.setText("\n".join(stats_lines))

    def _export_dxf(self):
        transformed = DATA_DIR / "output" / "transformed_parts.dxf"
        if not transformed.exists():
            # 尝试生成
            try:
                from dxf_processor import apply_graphic_transformations
                csv_p = DATA_DIR / "intermediate" / "transformation_info" / "placement_info.csv"
                src_dxf = DATA_DIR / "output" / "processed_parts.dxf"
                if csv_p.exists() and src_dxf.exists():
                    apply_graphic_transformations(str(csv_p), str(src_dxf), str(transformed))
            except Exception as e:
                msg_warn(self, "导出失败", f"生成 DXF 时出错:\n{e}")
                return

        if transformed.exists():
            dest, _ = QFileDialog.getSaveFileName(
                self, "保存 DXF", "arranged_parts.dxf", "DXF Files (*.dxf)")
            if dest:
                import shutil
                shutil.copy2(transformed, dest)
                msg_info(self, "导出成功", f"DXF 已保存至:\n{dest}")
        else:
            msg_warn(self, "文件不存在", "未找到 transformed_parts.dxf，请先运行排样计算。")

    def _export_image(self):
        viz = DATA_DIR / "output" / "substrate_visualization.png"
        if not viz.exists():
            # 尝试生成
            try:
                import subprocess as sp
                sp.run([sys.executable, str(SRC_DIR / "visualize_substrate.py")],
                       cwd=str(PROJECT_ROOT), timeout=30)
            except Exception:
                pass
        if viz.exists():
            dest, _ = QFileDialog.getSaveFileName(
                self, "保存图像", "substrate_result.png", "PNG Files (*.png)")
            if dest:
                import shutil
                shutil.copy2(viz, dest)
                msg_info(self, "导出成功", f"图像已保存至:\n{dest}")
        else:
            msg_warn(self, "文件不存在", "未找到基板可视化图。")


# ╔══════════════════════════════════════════════════════════════╗
# ║                      MAIN  WINDOW                            ║
# ╚══════════════════════════════════════════════════════════════╝

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Arrangeit — 不规则零件排样系统")
        self.setMinimumSize(900, 650)
        self.resize(1100, 750)

        # 中心 stacked widget
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # 3 个页面
        self.setup_page = SetupPage()
        self.config_page = ConfigPage()
        self.result_page = ResultPage()

        self.stack.addWidget(self.setup_page)    # index 0
        self.stack.addWidget(self.config_page)   # index 1
        self.stack.addWidget(self.result_page)   # index 2

        # ── 页面跳转信号 ──
        self.setup_page.dxf_processed.connect(self._on_dxf_processed)
        self.config_page.config_saved.connect(self._on_config_saved)
        self.result_page.restart_requested.connect(self._restart)

        # ── 状态变量 ──
        self.config = {}
        self.compute_worker = None

    # ── 页面跳转逻辑 ──

    def _on_dxf_processed(self, bitmap_dir: str):
        """DXF 处理完成 → 跳转到零件配置页"""
        self.config = self.setup_page.get_config()
        self.config_page.load_parts(bitmap_dir)
        self.stack.setCurrentIndex(1)

    def _on_config_saved(self, csv_path: str):
        """配置保存完成 → 运行排样计算 → 跳转结果页"""
        cfg = self.config
        # 切换到结果页并显示进度
        self.stack.setCurrentIndex(2)
        self.result_page.stats_label.setText("⏳ 正在运行排样计算...")
        self.result_page.viz_label.setText("计算中，请稍候...")

        self.compute_worker = ComputeWorker(
            substrate_w=cfg["substrate_w"],
            substrate_h=cfg["substrate_h"],
            pixel_res=cfg["pixel_res"],
            rotation_angles=ROTATION_OPTIONS,
            pg=cfg["cutting_gap"],
        )
        self.compute_worker.progress.connect(self._on_compute_log)
        self.compute_worker.finished_ok.connect(self._on_compute_done)
        self.compute_worker.start()

    def _on_compute_log(self, msg: str):
        current = self.result_page.stats_label.text()
        self.result_page.stats_label.setText(current + "\n" + msg)

    def _on_compute_done(self, success: bool, info: str):
        if success:
            self.result_page.show_results(info)
        else:
            self.result_page.stats_label.setText(f"❌ 排样计算失败:\n{info[-500:]}")
            msg_err(self, "计算失败", info[-500:])

    def _restart(self):
        """回到首页，彻底清理所有中间状态"""
        # 停止正在运行的后台线程
        if self.compute_worker and self.compute_worker.isRunning():
            self.compute_worker.terminate()
            self.compute_worker.wait(2000)
        self.compute_worker = None

        # 清空 ConfigPage 卡片与预览
        self.config_page.load_parts("")  # 空路径触发清理
        self.config_page._parts_all_path = ""
        self.config_page.preview_btn.setIcon(QPixmap())
        self.config_page.preview_btn.setText("")

        # 清空 ResultPage 展示
        self.result_page.viz_label.clear()
        self.result_page.viz_label.setText("")
        self.result_page.stats_label.setText("")

        # 重置 SetupPage 状态
        self.setup_page.bitmap_dir = ""
        self.setup_page.part_count = 0
        self.setup_page.log_area.setText("")
        self.setup_page.process_btn.setEnabled(True)
        self.setup_page.progress.setVisible(False)

        # 清空配置
        self.config = {}

        # 切回首页
        self.stack.setCurrentIndex(0)


# ╔══════════════════════════════════════════════════════════════╗
# ║                     STYLED  MSG  BOX                         ║
# ╚══════════════════════════════════════════════════════════════╝

_MSG_BOX_STYLE = """
    QMessageBox { background: #ffffff; }
    QMessageBox QLabel {
        color: #1a1a1a; font-size: 10pt;
    }
    QMessageBox QPushButton {
        min-width: 80px; min-height: 28px;
        background: #e0e0e0; color: #1a1a1a;
        border: 1px solid #a0a0a0; border-radius: 4px;
        padding: 5px 20px;
    }
    QMessageBox QPushButton:hover { background: #d0d0d0; }
"""

def msg_info(parent, title, text):
    box = QMessageBox(parent)
    box.setWindowTitle(title)
    box.setIcon(QMessageBox.Information)
    box.setText(text)
    box.setStyleSheet(_MSG_BOX_STYLE)
    box.exec()

def msg_warn(parent, title, text):
    box = QMessageBox(parent)
    box.setWindowTitle(title)
    box.setIcon(QMessageBox.Warning)
    box.setText(text)
    box.setStyleSheet(_MSG_BOX_STYLE)
    box.exec()

def msg_err(parent, title, text):
    box = QMessageBox(parent)
    box.setWindowTitle(title)
    box.setIcon(QMessageBox.Critical)
    box.setText(text)
    box.setStyleSheet(_MSG_BOX_STYLE)
    box.exec()


# ╔══════════════════════════════════════════════════════════════╗
# ║                        ENTRY  POINT                          ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    # 设置中文字体适配
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 全局样式 — 深色文字确保清晰可读
    app.setStyleSheet("""
        * { font-family: "Microsoft YaHei", "Segoe UI", sans-serif; }
        QMainWindow { background: #e8ecf0; }
        QWidget { color: #1a1a1a; }

        QLabel { color: #1a1a1a; }
        QGroupBox {
            font-size: 10pt; font-weight: bold;
            color: #1a1a1a;
            border: 1px solid #b0b0b0; border-radius: 6px;
            margin-top: 8px; padding-top: 18px;
            background: #ffffff;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px; padding: 0 6px;
            color: #2c3e50;
        }

        QLineEdit, QSpinBox, QDoubleSpinBox {
            border: 1px solid #b0b0b0; border-radius: 4px;
            padding: 4px 8px; background: #ffffff;
            color: #1a1a1a;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
            border: 1px solid #4a90d9;
        }

        QPushButton {
            border: 1px solid #a0a0a0; border-radius: 4px;
            padding: 6px 16px; background: #f0f0f0;
            color: #1a1a1a;
        }
        QPushButton:hover { background: #dcdcdc; border: 1px solid #888; }
        QPushButton:pressed { background: #c8c8c8; }

        QCheckBox { color: #1a1a1a; spacing: 6px; }
        QCheckBox::indicator { width: 16px; height: 16px; }

        QScrollArea { background: transparent; border: none; }
        QScrollBar:vertical { width: 10px; background: #e8ecf0; }
        QScrollBar::handle:vertical { background: #b0b0b0; border-radius: 5px; }
        QScrollBar::handle:vertical:hover { background: #888; }

        QProgressBar {
            border: 1px solid #b0b0b0; border-radius: 4px;
            text-align: center; color: #1a1a1a;
            background: #ffffff;
        }
        QProgressBar::chunk { background: #4a90d9; border-radius: 3px; }

        QMessageBox { background: #ffffff; }
        QMessageBox QLabel {
            color: #1a1a1a; font-size: 10pt;
        }
        QMessageBox QPushButton {
            min-width: 80px; min-height: 28px;
            background: #e0e0e0; color: #1a1a1a;
            border: 1px solid #a0a0a0; border-radius: 4px;
            padding: 5px 20px;
        }
        QMessageBox QPushButton:hover { background: #d0d0d0; }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
