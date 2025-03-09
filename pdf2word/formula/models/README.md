# 数学公式处理模型

此目录用于存放数学公式检测和识别所需的模型文件。

## 所需模型

1. **公式检测模型 (formula_detection_yolov8.pt)**
   - 基于YOLOv8的数学公式检测模型
   - 用于在PDF页面中定位数学公式区域

2. **公式识别模型 (unimernet_formula_recognition/)**
   - 基于UnimerNet的数学公式识别模型
   - 用于将检测到的公式区域转换为LaTeX格式

## 自动查找通过download_models_hf.py下载的模型

如果您已经使用`download_models_hf.py`脚本下载了模型，系统会自动尝试从以下路径查找模型：

1. 检查用户主目录下的`magic-pdf.json`配置文件
2. 从中获取`models-dir`设置，该目录下应该包含以下子目录：
   - `MFD/YOLO/` - 包含数学公式检测模型
   - `MFR/unimernet_small_2501/` - 包含数学公式识别模型

系统会自动搜索这些目录及其子目录中的模型文件，您无需手动配置路径。

## 解决常见问题

### 1. 模型文件不兼容问题

如果您遇到了`Error no file named pytorch_model.bin, model.safetensors...`之类的错误，表示通过`download_models_hf.py`下载的模型文件结构与标准Hugging Face模型不完全兼容。可以使用修复脚本：

```bash
python -m pdf2word.formula.fix_unimernet_model
```

### 2. "You need to specify either 'text' or 'text_target'" 错误

如果您遇到此错误，这是UnimerNet模型的配置问题。我们提供了专门的补丁脚本来修复：

```bash
python -m pdf2word.formula.patch_unimernet --update-config
```

此脚本会：
1. 为模型添加必要的配置和支持文件
2. 创建处理器配置，包含必要的文本配置
3. 自动更新系统配置，使用修补后的模型

### 3. 自定义解决方法

如果以上方法不起作用，您可以尝试手动创建/修改必要的配置文件：

```bash
# 查看模型文件和目录
ls -la ~/.cache/modelscope/hub/models/opendatalab/PDF-Extract-Kit-1___0/models/MFR/unimernet_small_2501/

# 创建新的模型目录
mkdir -p ~/unimernet_model_fixed

# 复制所有现有文件
cp ~/.cache/modelscope/hub/models/opendatalab/PDF-Extract-Kit-1___0/models/MFR/unimernet_small_2501/* ~/unimernet_model_fixed/

# 重命名模型文件
cp ~/.cache/modelscope/hub/models/opendatalab/PDF-Extract-Kit-1___0/models/MFR/unimernet_small_2501/pytorch_model.pth ~/unimernet_model_fixed/pytorch_model.bin

# 创建必要的配置文件，包含text_config
echo '{
  "processor_class": "VisionTextDualEncoderProcessor",
  "text_config": {
    "model_max_length": 512,
    "tokenizer_class": "PreTrainedTokenizer"
  }
}' > ~/unimernet_model_fixed/preprocessor_config.json

# 更新配置
python -m pdf2word.formula.update_model_paths --recognizer=~/unimernet_model_fixed
```

## 模型下载

您可以通过以下两种方式之一获取所需模型：

### 方式一：使用`download_models_hf.py`脚本

这将从Hugging Face下载所有必要的模型，并自动配置：

```bash
python download_models_hf.py
```

### 方式二：手动下载单个模型

如果您只想使用公式处理功能，可以从以下来源手动下载模型：

- YOLOv8公式检测模型: 
  - Hugging Face: https://huggingface.co/models?search=math+formula+detection
  - 或通过ultralytics训练自定义模型: https://github.com/ultralytics/ultralytics

- UnimerNet公式识别模型: 
  - Hugging Face: https://huggingface.co/models?search=math+formula+recognition

## 手动配置模型路径

如果自动查找失败，或者您想使用自定义位置的模型，可以通过以下方式指定路径：

1. 使用配置更新脚本：
   ```bash
   python -m pdf2word.formula.update_model_paths --detector=/path/to/your/formula_detection_yolov8.pt --recognizer=/path/to/your/unimernet_formula_recognition
   ```

2. 或者在命令行直接指定路径（临时使用）：
   ```bash
   python -m pdf2word.main convert input.pdf output.docx --process_formulas=True --formula_detector_model=/path/to/your/detector.pt --formula_recognizer_model=/path/to/your/recognizer
   ```

## 模型安装

1. 下载模型文件后，将它们放置在此目录中：
   ```
   pdf2word/formula/models/formula_detection_yolov8.pt
   pdf2word/formula/models/unimernet_formula_recognition/
   ```

2. 确保已安装所有必要的依赖项：
   ```
   pip install -r requirements.txt
   ```

## 自定义模型

您也可以通过命令行参数指定自己的自定义模型路径：

```bash
python -m pdf2word.main convert input.pdf output.docx --process_formulas=True --formula_detector_model=/path/to/your/detector.pt --formula_recognizer_model=/path/to/your/recognizer
``` 