# RapidOCR 视频硬字幕提取器

基于 `RapidOCR` + `OpenCV` + `PySide6` 的桌面工具：从视频画面中的硬字幕区域（ROI）识别文字并导出 `SRT` 字幕文件。
写这个软件主要是为了解决mac端没有gui版，并且用的是CPU OCR，时间较长。win平台有更好用的开源软件。比如：https://github.com/YaoFANGUK/video-subtitle-extractor 和 https://github.com/timminator/VideOCR
这两个都是采用 PaddleOCR引擎，目测比RapidOCR引擎效果好。本来打算用PaddleOCR，各种报错。因为没有arm 版 mac，所以没有arm 版软件包。可用pyinstaller自行打包。

## 功能

- 选择视频（`mp4/mkv/avi/mov`）
- 拖动/缩放绿色框选择字幕区域（ROI）
- 时间轴拖动预览帧
- 进度条与剩余时间估算
- 一键取消任务
- 默认保存到视频同目录：`<视频名>.srt`（也可手动改保存路径）

## 环境要求

- Python 3.8+
- 可正常安装 `PySide6` 与 `opencv-python` 的系统环境（Windows/macOS/Linux 均可）

## 安装依赖

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -U pip
pip install PySide6 opencv-python rapidocr_onnxruntime
```

## 运行

```bash
python main.py
```

## 使用流程

1. 点击「添加视频」选择视频文件
2. 在预览画面中拖动/缩放绿色框，让其覆盖字幕区域
3. （可选）点击「设置保存目录」或直接在输入框里修改输出路径
4. 点击「确定 / 开始提取」
5. 完成后会弹窗提示保存位置

## 输出说明

- 输出文件：`SRT`，UTF-8 编码
- 会过滤持续时间小于 `100ms` 的极短闪烁误识别片段
- 识别逻辑为“采样帧 OCR + 文本变化分段”，对复杂字幕（多行、特效字、滚动字幕）可能需要更精细的采样或更严格的 ROI

## 可调参数（在 `main.py` 中修改）

- 采样间隔：`OCRWorker.run()` 内的 `skip_frames = 5`（越小越精细但更慢）
- 置信度阈值：`OCRWorker(..., threshold=0.6)`（越高越严格，漏字风险更大）
- 默认 ROI 框位置：`VideoLabel.__init__()` 里的 `self.roi_rect = QRect(...)`

## 打包（可选）

项目内已包含 `硬字幕提取器.spec`，可用 PyInstaller 打包：

```bash
pip install pyinstaller
pyinstaller 硬字幕提取器.spec
```

