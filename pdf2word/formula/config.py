'''Configuration for formula processing module.'''
import os
import json
from pathlib import Path
import logging
import glob

# 获取项目根目录（可根据实际情况调整）
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 尝试读取magic-pdf.json配置文件
def get_model_paths_from_magic_pdf():
    """尝试从magic-pdf.json配置文件中获取模型路径"""
    home_dir = os.path.expanduser('~')
    magic_pdf_config = os.path.join(home_dir, 'magic-pdf.json')
    
    if os.path.exists(magic_pdf_config):
        try:
            with open(magic_pdf_config, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            models_dir = config.get('models-dir')
            if models_dir:
                # 检查MFD目录是否存在
                mfd_base_dir = os.path.join(models_dir, 'MFD')
                if os.path.exists(mfd_base_dir):
                    formula_detector = find_model_directory(mfd_base_dir, 'YOLO')
                else:
                    formula_detector = None
                
                # 检查MFR目录是否存在
                mfr_base_dir = os.path.join(models_dir, 'MFR')
                if os.path.exists(mfr_base_dir):
                    formula_recognizer = find_model_directory(mfr_base_dir, 'unimernet_small_2501')
                else:
                    formula_recognizer = None
                
                # 至少有一个模型路径有效，则返回配置
                if formula_detector or formula_recognizer:
                    result = {}
                    if formula_detector:
                        result['FORMULA_DETECTOR'] = formula_detector
                    if formula_recognizer:
                        result['FORMULA_RECOGNIZER'] = formula_recognizer
                    return result
                
            # 尝试搜索缓存目录
            cache_dir = os.path.join(home_dir, '.cache')
            if os.path.exists(cache_dir):
                model_paths = find_models_in_cache(cache_dir)
                if model_paths:
                    return model_paths
        except Exception as e:
            logging.warning(f"无法读取magic-pdf.json: {e}")
    
    return None

def find_model_directory(base_dir, dir_name):
    """找到特定名称的模型目录，如果找不到，尝试匹配部分名称"""
    # 直接检查精确名称
    target_dir = os.path.join(base_dir, dir_name)
    if os.path.exists(target_dir):
        return target_dir
    
    # 如果找不到精确名称，尝试查找包含该名称的目录
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and dir_name.lower() in item.lower():
            return item_path
    
    # 如果还是找不到，返回base_dir本身
    return base_dir

def find_models_in_cache(cache_dir):
    """在缓存目录中查找可能的模型文件"""
    result = {}
    
    # 尝试在缓存中查找YOLOv8相关文件
    yolo_files = glob.glob(os.path.join(cache_dir, '**', '*yolo*.pt'), recursive=True)
    if yolo_files:
        # 使用找到的第一个YOLO模型
        result['FORMULA_DETECTOR'] = yolo_files[0]
    
    # 查找可能的公式识别模型目录
    model_dirs = []
    
    # 常见的公式识别模型目录名称部分
    recognizer_names = ['unimernet', 'formula', 'math', 'latex', 'mfr']
    
    # 首先在.cache/modelscope目录下查找
    modelscope_dir = os.path.join(cache_dir, 'modelscope')
    if os.path.exists(modelscope_dir):
        for name in recognizer_names:
            # 递归查找包含特定名称的目录
            for root, dirs, _ in os.walk(modelscope_dir):
                for d in dirs:
                    if name in d.lower():
                        model_dirs.append(os.path.join(root, d))
    
    # 然后在.cache/huggingface目录下查找
    huggingface_dir = os.path.join(cache_dir, 'huggingface')
    if os.path.exists(huggingface_dir):
        for name in recognizer_names:
            # 递归查找包含特定名称的目录
            for root, dirs, _ in os.walk(huggingface_dir):
                for d in dirs:
                    if name in d.lower():
                        model_dirs.append(os.path.join(root, d))
    
    # 如果找到了可能的识别模型目录，使用第一个
    if model_dirs:
        result['FORMULA_RECOGNIZER'] = model_dirs[0]
    
    return result if result else None

# 尝试从magic-pdf.json获取路径
magic_pdf_models = get_model_paths_from_magic_pdf()
if magic_pdf_models:
    for model_type, path in magic_pdf_models.items():
        logging.info(f"Found {model_type} model at: {path}")

# 默认模型路径配置
MODELS = {
    # YOLOv8公式检测模型路径
    'FORMULA_DETECTOR': str(PROJECT_ROOT / 'models' / 'formula_detection_yolov8.pt'),
    
    # UnimerNet公式识别模型路径
    'FORMULA_RECOGNIZER': '/Users/tibelf/.cache/modelscope/hub/models/opendatalab/PDF-Extract-Kit-1___0/models/MFR/unimernet_patched',
}

# 如果找到了magic-pdf.json中的模型，则更新默认配置
if magic_pdf_models:
    MODELS.update(magic_pdf_models)

def get_model_path(model_type):
    '''Get the path for a specific model type.
    
    Args:
        model_type (str): Type of model ('FORMULA_DETECTOR' or 'FORMULA_RECOGNIZER')
        
    Returns:
        str: Path to the model
    '''
    return MODELS.get(model_type) 