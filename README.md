# Arrangeit — 不规则零件自动排样系统

为激光切割 / 铣削工艺中的不规则机械零件提供自动化排样（packing）方案。将 CAD 图纸中的零件识别、二值化后，基于归一化互相关（NCC）模板匹配 + 贪心策略在基板上自动排列，最后输出可直接用于加工的 DXF 文件。

> 参考：Meng, L., Ding, L., Pu, Y. et al. *Optimizing 2D irregular packing via image processing and computational intelligence.* Sci Rep 15, 12320 (2025). https://doi.org/10.1038/s41598-025-97202-0

---

## 目录结构

```
Arrangeit/
├── src/                          # 核心源码
│   ├── gui_app.py                # PySide6 图形界面入口
│   ├── main.py                   # CLI 主控脚本 + 全局配置
│   ├── dxf_processor.py          # DXF 解析、零件识别、位图生成、图形变换
│   ├── computation_core.py       # 排样计算核心（NCC + 贪心放置）
│   └── visualize_substrate.py    # 基板占用可视化
│
├── module/                       # 早期试验脚本（ezdxf 验证用，非主流程）
│   ├── trial3.py / trial4.py / trial5.py
│
├── data/
│   ├── input/                    # 输入：completeINPUT.dxf（用户放置的待处理图纸）
│   ├── intermediate/             # 中间产物（每次运行自动清理再生）
│   │   ├── template_matrix/      # 每个零件的 .npy 二值矩阵
│   │   ├── user_input/           # input_data.csv（副本数 + 角度配置）
│   │   ├── transformation_info/  # placement_info.csv（排样结果坐标）
│   │   ├── final_substrate/      # final_substrate.npy（最终占用矩阵）
│   │   └── run_config.json       # GUI 传入计算核心的运行参数
│   └── output/                   # 最终产物
│       ├── bitmaps_offset_pgX.X/ # 每个零件的偏移位图 PNG
│       ├── parts_all.png         # 全部零件编号总览图
│       ├── parts_outer_only.png  # 仅外轮廓图
│       ├── processed_parts.dxf   # 零件打包为 block 的中间 DXF
│       ├── transformed_parts.dxf # 排样变换后的最终 DXF
│       └── substrate_*.png       # 基板可视化图
│
├── run_gui.bat                   # Windows 双击启动 GUI
└── README.md
```

---

## 核心模块

### `dxf_processor.py` — DXF 处理与零件识别

- 使用 **ezdxf** 解析 DXF，支持 LINE / ARC / POLYLINE / LWPOLYLINE / SPLINE / CIRCLE / ELLIPSE 等实体
- 利用 **shapely** 的封闭曲线包含关系判断将外轮廓与内孔归组为同一零件
- 每个零件打包为独立的 **BLOCK**，存储于 `processed_parts.dxf`
- 生成每个零件的位图 PNG（带 `pg` 切割间隙的形态学偏移），同时输出 `.npy` 二值矩阵
- 提供 `apply_graphic_transformations()` 读取排样结果 CSV，对 block 进行旋转+平移，输出最终 DXF

### `computation_core.py` — 排样计算核心

1. 从 `template_matrix/` 加载每个零件的二值矩阵，按照 `input_data.csv` 中的副本数→角度组合预旋转展开
2. 对每个模板在基板上做**滑动窗口 NCC（归一化互相关）**计算
3. **贪心放置**：每个副本尝试其所有允许角度，选取 NCC 匹配度最高者放置（平局选较小角度）
4. 放置后更新基板占用状态（像素级 OR 操作），继续下一个副本
5. 输出 `placement_info.csv`（零件序号、副本序号、最终角度、中心点坐标）和 `final_substrate.npy`

### `main.py` — CLI 编排器

- 定义全局参数（基板尺寸、像素精度、切割间隙、旋转角度）
- 串联 DXF 处理 → 基板创建 → 排样计算 → 图形变换 4 步流程
- 被 `dxf_processor` 和 `computation_core` 导入共享配置

### `gui_app.py` — PySide6 图形界面

3 步向导式操作，详见下方「GUI 操作」。

### `visualize_substrate.py` — 可视化

- 读取 `final_substrate.npy`，用 matplotlib 渲染为黑白占位图

---

## 关键概念

| 参数 | 含义 |
|---|---|
| **副本数** | 该零件一共需要几个。比如设为 3 → 最终输出 3 个该零件 |
| **旋转角度** | 每个副本**允许**以哪些角度尝试放置。勾选 0°+90° → 程序为每个副本尝试这两个角度，自动保留匹配度最高的那个 |
| **切割间隙 pg** | 零件之间的安全间距 (mm)。在位图生成阶段通过形态学偏移实现 |
| **像素精度** | 多少毫米对应 1 像素。越小越精确，但计算越慢 |

> ⚠️ 副本数 ≠ 副本数 × 角度数。角度只是「允许尝试的方向」，不是「每个角度都出一个副本」。

---

## 数据流

### CLI 路径

```
completeINPUT.dxf
       │
       │  [dxf_processor.py]
       ▼
  ┌─────────────────────┐
  │ 零件识别 → BLOCK     │──→ processed_parts.dxf
  │ 位图渲染 + 偏移      │──→ bitmaps_offset_pgX.X/part_*.png
  │ 二值矩阵输出         │──→ template_matrix/part_*.npy
  │ 总览图输出           │──→ parts_all.png
  └─────────────────────┘
       │
       │  用户手动编辑 input_data.csv
       ▼
  ┌─────────────────────┐
  │ [computation_core]  │
  │ NCC模板匹配          │
  │ 贪心排样             │
  └─────────────────────┘
       │
       ├──→ placement_info.csv     (排样结果坐标)
       ├──→ final_substrate.npy   (基板占用矩阵)
       │
       ▼
  [visualize_substrate.py]  ──→ substrate_visualization.png
       │
       │  [dxf_processor: apply_graphic_transformations]
       ▼
  transformed_parts.dxf   ←── 最终可加工的排样 DXF
```

### GUI 路径

```
┌────────────────────────────────────────────────┐
│  Step 1 — 输入设置                               │
│  · 浏览选择 DXF 文件                              │
│  · 设置基板宽/高、像素精度、切割间隙               │
│  · 点击 [处理 DXF 文件]                           │
│       │                                          │
│       │ 后台线程: 清理旧数据 → dxf_processor       │
│       ▼                                          │
│  位图生成完毕，自动跳转 ──────────────────────────│
├────────────────────────────────────────────────┤
│  Step 2 — 配置零件                               │
│  · 顶部: 零件总览预览图（可点击放大）              │
│  · 批量操作面板:                                   │
│    ☑ 批量设置副本数 + 输入框 + [应用到全部]        │
│    批量设置角度 ☑0°☑90°☐180°☐270° [应用到全部]   │
│  · 下方: 每个零件一张卡片                          │
│    ┌──────────┐                                  │
│    │ 零件 #0   │                                  │
│    │ [缩略图]  │                                  │
│    │ 副本: [3] │                                  │
│    │ ☑0°☑90°  │                                  │
│    │ ☑180°☑270°│                                 │
│    └──────────┘                                  │
│  · 点击 [运行排样计算]                             │
│       │  写入 input_data.csv                      │
│       │  切换至结果页，启动 ComputeWorker           │
│       ▼                                          │
├────────────────────────────────────────────────┤
│  Step 3 — 排样结果                               │
│  · 后台:                                          │
│    写入 run_config.json                           │
│    → 子进程运行 computation_core.py               │
│    → 子进程运行 visualize_substrate.py            │
│  · 显示基板可视化图                                │
│  · 统计摘要（放置数 / 利用率）                     │
│  · [导出 DXF] [导出图像] [重新开始]               │
└────────────────────────────────────────────────┘
```

**GUI 数据流要点：**

- 每次点击「处理 DXF 文件」→ 自动清理上一次的全部中间/输出数据，保证新旧不混淆
- `computation_core.py` 通过子进程运行；GUI 将配置写入 `run_config.json`，计算核心优先读取 JSON 而非 `main.py` 默认值
- 点击「重新开始」→ 断开后台线程、清空三页 UI 状态、回到首页

---

## 快速开始

### 环境要求

```
Python 3.10+
pip install PySide6 ezdxf shapely numpy matplotlib Pillow
```

### CLI 方式

```bash
# 1. 将 DXF 图纸放入 data/input/completeINPUT.dxf
# 2. 手动编辑 data/intermediate/user_input/input_data.csv
# 3. 运行
python src/main.py --input data/input/completeINPUT.dxf
```

### GUI 方式（推荐）

```bash
# Windows: 双击
run_gui.bat

# 或命令行
python src/gui_app.py
```

---

## 依赖一览

| 库 | 用途 |
|---|---|
| `PySide6` | GUI 界面 |
| `ezdxf` | DXF 文件读写与实体解析 |
| `shapely` | 几何包含关系判断（零件识别） |
| `numpy` | 矩阵运算、NCC 模板匹配 |
| `matplotlib` | 位图渲染、基板可视化 |
| `Pillow` | 图像填充与 PNG 写入 |

---

## 参考文献

Meng, L., Ding, L., Pu, Y. et al. *Optimizing 2D irregular packing via image processing and computational intelligence.* Sci Rep 15, 12320 (2025). https://doi.org/10.1038/s41598-025-97202-0
