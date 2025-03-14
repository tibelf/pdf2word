# pdf2word

一个基于开源项目 [pdf2docx](https://github.com/dothinking/pdf2docx) 的二次开发工具，提供高质量的 PDF 转 Word 文档功能。

## 功能特性

- 基于 `PyMuPDF` 提取文本、图片、矢量等原始数据 
- 基于规则解析章节、段落、表格、图片、文本等布局及样式
- 基于 `python-docx` 创建Word文档
- 纯命令行界面，适合批处理和自动化流程
- 支持 conda 环境管理

## 主要功能

- 解析和创建页面布局
    - 页边距
    - 章节和分栏 (最多支持两栏布局)

- 解析和创建段落
    - 水平（从左到右）或竖直（自底向上）方向文本
    - 字体样式例如字体、字号、粗/斜体、颜色
    - 文本样式例如高亮、下划线和删除线
    - 段落水平对齐方式 (左/右/居中/分散对齐)及前后间距
    
- 解析和创建图片
	- 内联图片
    - 灰度/RGB/CMYK等颜色空间图片
    - 带有透明通道图片
    - 浮动图片（衬于文字下方）

- 解析和创建表格
    - 边框样式例如宽度和颜色
    - 单元格背景色
    - 合并单元格
    - 单元格垂直文本
    - 隐藏部分边框线的表格
    - 嵌套表格

- 支持多进程转换

## 项目结构

```
pdf2word/
├── pdf2word/           # 主要代码目录
│   ├── __init__.py     # 包初始化文件
│   ├── converter.py    # 核心转换器
│   ├── main.py         # 命令行入口
│   ├── common/         # 通用工具
│   ├── font/           # 字体处理
│   ├── image/          # 图像处理
│   ├── layout/         # 布局处理
│   ├── page/           # 页面处理
│   ├── shape/          # 形状处理
│   ├── table/          # 表格处理
│   └── text/           # 文本处理
├── examples/           # 使用示例
├── environment.yml     # Conda环境配置
├── requirements.txt    # pip依赖项
├── setup.py            # 安装配置
└── start.sh            # 启动脚本
```

## 快速开始

1. 克隆仓库

```bash
git clone https://github.com/yourusername/pdf2word.git
cd pdf2word
```

2. 使用启动脚本转换PDF（自动设置Conda环境）

```bash
./start.sh your_file.pdf output.docx
```

3. 或者手动安装并使用

```bash
# 创建Conda环境
conda env create -f environment.yml

# 激活环境
conda activate pdf2word

# 安装pdf2word
pip install -e .

# 使用CLI转换
pdf2word convert your_file.pdf output.docx
```

4. 查看更多示例

```bash
cd examples
python test_convert.py your_file.pdf
```

## 使用方法

1. 安装 conda 环境（推荐）：

```bash
conda env create -f environment.yml
```

2. 使用 start.sh 脚本运行转换：

```bash
./start.sh input.pdf output.docx
```

或者指定页码范围：

```bash
./start.sh input.pdf output.docx --start=1 --end=10
```

## 限制

- 目前暂不支持扫描PDF文字识别
- 仅支持从左向右书写的语言
- 不支持旋转的文字
- 基于规则的解析无法保证100%还原PDF样式

## 致谢

本项目基于 [pdf2docx](https://github.com/dothinking/pdf2docx) 开发，特此感谢原作者的杰出工作。
