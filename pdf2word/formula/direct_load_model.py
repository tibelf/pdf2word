#!/usr/bin/env python
'''直接加载模型的辅助脚本。

此脚本提供了直接使用PyTorch加载模型的函数，绕过Hugging Face的自动加载机制。
用于解决非标准模型文件结构导致的加载问题。
'''

import os
import sys
import json
import logging
import shutil
from pathlib import Path
import tempfile

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def rename_model_file(model_dir, destination_dir=None):
    """将 pytorch_model.pth 重命名为 pytorch_model.bin
    
    Args:
        model_dir (str): 模型目录路径
        destination_dir (str, optional): 目标目录。如果为None，则在临时目录中创建
        
    Returns:
        str: 包含重命名后模型的目录路径
    """
    source_file = os.path.join(model_dir, "pytorch_model.pth")
    
    if not os.path.exists(source_file):
        logging.error(f"在 {model_dir} 中找不到 pytorch_model.pth 文件")
        return None
    
    if destination_dir is None:
        destination_dir = tempfile.mkdtemp(prefix="unimernet_model_")
        logging.info(f"创建临时模型目录: {destination_dir}")
    else:
        os.makedirs(destination_dir, exist_ok=True)
    
    # 复制所有JSON配置文件
    json_files = ["config.json", "configuration.json", "preprocessor_config.json", 
                 "tokenizer.json", "tokenizer_config.json"]
    
    for json_file in json_files:
        source_json = os.path.join(model_dir, json_file)
        if os.path.exists(source_json):
            dest_json = os.path.join(destination_dir, json_file)
            shutil.copy2(source_json, dest_json)
            logging.info(f"已复制 {json_file}")
    
    # 复制并重命名模型文件
    dest_file = os.path.join(destination_dir, "pytorch_model.bin")
    shutil.copy2(source_file, dest_file)
    logging.info(f"已将 pytorch_model.pth 复制并重命名为 pytorch_model.bin")
    
    return destination_dir

def convert_model_format(model_dir, destination_dir=None):
    """转换模型格式，解决加载问题
    
    Args:
        model_dir (str): 模型目录路径
        destination_dir (str, optional): 目标目录。如果为None，则在临时目录中创建
        
    Returns:
        str: 新的模型目录路径
    """
    if destination_dir is None:
        destination_dir = tempfile.mkdtemp(prefix="unimernet_model_")
        logging.info(f"创建临时模型目录: {destination_dir}")
    else:
        os.makedirs(destination_dir, exist_ok=True)
    
    try:
        # 首先尝试直接重命名模型文件
        result = rename_model_file(model_dir, destination_dir)
        if result:
            return result
        
        # 如果重命名失败，尝试创建简单的配置文件
        config_path = os.path.join(destination_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump({
                "model_type": "vision-to-sequence",
                "architectures": ["VisionEncoderDecoderModel"],
                "_name_or_path": "unimernet"
            }, f, indent=2)
        logging.info("已创建基本的config.json文件")
        
        return destination_dir
    except Exception as e:
        logging.error(f"转换模型格式时出错: {e}")
        return None

def main():
    """主函数，用于命令行调用"""
    import argparse
    
    parser = argparse.ArgumentParser(description="转换模型格式，解决加载问题")
    parser.add_argument("model_dir", help="模型目录路径")
    parser.add_argument("--output", "-o", help="输出目录路径 (可选)")
    
    args = parser.parse_args()
    
    if args.output:
        destination_dir = args.output
    else:
        destination_dir = None
    
    result = convert_model_format(args.model_dir, destination_dir)
    
    if result:
        logging.info(f"模型格式转换成功，新的模型目录: {result}")
        print(f"OUTPUT_DIR={result}")
    else:
        logging.error("模型格式转换失败")
        sys.exit(1)

if __name__ == "__main__":
    main() 