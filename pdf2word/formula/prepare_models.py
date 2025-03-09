#!/usr/bin/env python
'''准备模型文件的辅助脚本。

此脚本可以帮助用户检测和准备公式处理所需的模型文件。
它会检测可能的模型路径，并从中拷贝出有用的文件组织成标准结构。
'''

import os
import sys
import shutil
import logging
import glob
import json
from pathlib import Path
import tempfile

# 设置日志
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

# 从配置文件中导入函数
from pdf2word.formula.config import get_model_paths_from_magic_pdf, MODELS

def find_all_model_files(directory, file_types=None):
    """递归查找目录中的所有模型文件"""
    if file_types is None:
        file_types = [
            "*.bin", "*.safetensors", "*.h5", "*.ckpt.*", "*.msgpack",
            "config.json", "*.json", "tokenizer.json", "vocab.txt",
            "*.pt", "*.pth"
        ]
    
    all_files = []
    for file_type in file_types:
        all_files.extend(glob.glob(os.path.join(directory, "**", file_type), recursive=True))
    
    return all_files

def prepare_detector_model(source_dir, target_dir):
    """准备检测器模型"""
    if not os.path.exists(source_dir):
        logging.error(f"源目录不存在: {source_dir}")
        return False
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 查找.pt或.pth文件
    model_files = glob.glob(os.path.join(source_dir, "**", "*.pt"), recursive=True)
    model_files.extend(glob.glob(os.path.join(source_dir, "**", "*.pth"), recursive=True))
    
    if not model_files:
        logging.error(f"在 {source_dir} 中找不到检测器模型文件 (.pt 或 .pth)")
        return False
    
    # 使用第一个找到的文件
    model_file = model_files[0]
    target_file = os.path.join(target_dir, os.path.basename(model_file))
    
    try:
        shutil.copy2(model_file, target_file)
        logging.info(f"已复制检测器模型文件: {model_file} -> {target_file}")
        return True
    except Exception as e:
        logging.error(f"复制检测器模型时出错: {e}")
        return False

def prepare_recognizer_model(source_dir, target_dir):
    """准备识别器模型"""
    if not os.path.exists(source_dir):
        logging.error(f"源目录不存在: {source_dir}")
        return False
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 查找所有可能的模型文件
    model_files = find_all_model_files(source_dir)
    
    if not model_files:
        logging.error(f"在 {source_dir} 中找不到任何模型文件")
        return False
    
    success = False
    for model_file in model_files:
        rel_path = os.path.relpath(model_file, source_dir)
        target_file = os.path.join(target_dir, rel_path)
        
        # 确保目标目录存在
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        
        try:
            shutil.copy2(model_file, target_file)
            logging.info(f"已复制识别器模型文件: {rel_path}")
            success = True
        except Exception as e:
            logging.warning(f"复制文件时出错 {model_file}: {e}")
    
    if success:
        # 创建一个基本的config.json，如果不存在
        config_path = os.path.join(target_dir, "config.json")
        if not os.path.exists(config_path):
            try:
                with open(config_path, "w") as f:
                    json.dump({
                        "model_type": "vision-to-sequence",
                        "architectures": ["VisionEncoderDecoderModel"],
                        "_name_or_path": "unimernet"
                    }, f, indent=2)
                logging.info("已创建基本的config.json文件")
            except Exception as e:
                logging.warning(f"创建config.json时出错: {e}")
    
    return success

def main():
    """主函数"""
    logging.info("开始检测模型路径...")
    
    # 获取模型路径
    magic_pdf_models = get_model_paths_from_magic_pdf()
    
    if magic_pdf_models:
        for model_type, path in magic_pdf_models.items():
            logging.info(f"找到 {model_type} 模型: {path}")
    else:
        logging.warning("从magic-pdf.json中未找到模型路径")
    
    # 获取当前配置的模型路径
    logging.info("\n当前配置:")
    for model_type, path in MODELS.items():
        if os.path.exists(path):
            logging.info(f"{model_type}: {path} [存在]")
        else:
            logging.info(f"{model_type}: {path} [不存在]")
    
    # 询问用户是否准备模型
    prepare = input("\n是否要准备模型文件到标准位置? (y/n): ").lower() in ['y', 'yes']
    
    if not prepare:
        logging.info("已取消准备模型")
        return
    
    # 准备目标目录
    formula_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(formula_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    detector_dir = os.path.join(models_dir, "formula_detection_yolov8")
    recognizer_dir = os.path.join(models_dir, "unimernet_formula_recognition")
    
    # 清空目标目录
    for dir_path in [detector_dir, recognizer_dir]:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                logging.info(f"已清空目录: {dir_path}")
            except Exception as e:
                logging.error(f"清空目录 {dir_path} 时出错: {e}")
    
    os.makedirs(detector_dir, exist_ok=True)
    os.makedirs(recognizer_dir, exist_ok=True)
    
    # 准备检测器模型
    detector_source = magic_pdf_models.get('FORMULA_DETECTOR') if magic_pdf_models else None
    if detector_source:
        logging.info(f"开始准备检测器模型: {detector_source} -> {detector_dir}")
        if prepare_detector_model(detector_source, detector_dir):
            logging.info("检测器模型准备成功")
        else:
            logging.error("检测器模型准备失败")
    else:
        logging.warning("未找到检测器模型源")
    
    # 准备识别器模型
    recognizer_source = magic_pdf_models.get('FORMULA_RECOGNIZER') if magic_pdf_models else None
    if recognizer_source:
        logging.info(f"开始准备识别器模型: {recognizer_source} -> {recognizer_dir}")
        if prepare_recognizer_model(recognizer_source, recognizer_dir):
            logging.info("识别器模型准备成功")
        else:
            logging.error("识别器模型准备失败")
    else:
        logging.warning("未找到识别器模型源")
    
    logging.info("\n模型准备完成，请尝试使用以下路径:")
    logging.info(f"检测器模型: {detector_dir}")
    logging.info(f"识别器模型: {recognizer_dir}")

if __name__ == "__main__":
    main() 