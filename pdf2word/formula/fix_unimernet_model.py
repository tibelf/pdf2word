#!/usr/bin/env python
'''快速修复UnimerNet模型格式问题的脚本。

此脚本专门用于解决"Error no file named pytorch_model.bin"错误，
通过将pytorch_model.pth重命名为pytorch_model.bin并创建必要的配置文件。
'''

import os
import sys
import shutil
import logging
import json
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# 查找特定路径中的模型
DEFAULT_MODEL_PATH = os.path.expanduser("~/.cache/modelscope/hub/models/opendatalab/PDF-Extract-Kit-1___0/models/MFR/unimernet_small_2501")

def fix_unimernet_model(model_path=DEFAULT_MODEL_PATH, output_dir=None):
    """修复UnimerNet模型格式问题
    
    Args:
        model_path (str): 模型路径，默认为常见的下载位置
        output_dir (str, optional): 输出目录，如果不指定则在模型目录旁创建
        
    Returns:
        str: 修复后的模型目录路径
    """
    if not os.path.exists(model_path):
        logging.error(f"模型路径不存在: {model_path}")
        return None
    
    # 检查是否有pytorch_model.pth文件
    pth_file = os.path.join(model_path, "pytorch_model.pth")
    if not os.path.exists(pth_file):
        logging.error(f"找不到pytorch_model.pth文件: {pth_file}")
        return None
    
    # 确定输出目录
    if output_dir is None:
        parent_dir = os.path.dirname(model_path)
        output_dir = os.path.join(parent_dir, "unimernet_fixed")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"将修复后的模型保存到: {output_dir}")
    
    # 复制所有JSON文件
    json_files = ["config.json", "configuration.json", "preprocessor_config.json", 
                 "tokenizer.json", "tokenizer_config.json"]
    
    for json_file in json_files:
        source_json = os.path.join(model_path, json_file)
        if os.path.exists(source_json):
            dest_json = os.path.join(output_dir, json_file)
            shutil.copy2(source_json, dest_json)
            logging.info(f"已复制 {json_file}")
    
    # 复制并重命名模型文件
    dest_file = os.path.join(output_dir, "pytorch_model.bin")
    shutil.copy2(pth_file, dest_file)
    logging.info(f"已将 pytorch_model.pth 复制并重命名为 pytorch_model.bin")
    
    # 确保存在config.json
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_path):
        try:
            with open(config_path, "w") as f:
                json.dump({
                    "model_type": "vision-to-sequence",
                    "architectures": ["VisionEncoderDecoderModel"],
                    "_name_or_path": "unimernet",
                    "attention_probs_dropout_prob": 0.1,
                    "hidden_dropout_prob": 0.1,
                    "hidden_size": 768,
                    "intermediate_size": 3072,
                    "layer_norm_eps": 1e-12,
                    "max_position_embeddings": 512,
                    "num_attention_heads": 12,
                    "num_hidden_layers": 12,
                    "vocab_size": 30522
                }, f, indent=2)
            logging.info("已创建config.json配置文件")
        except Exception as e:
            logging.warning(f"创建config.json时出错: {e}")
    
    # 创建一个简单的README说明修复过程
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"""# 修复后的UnimerNet模型

此目录包含修复后的UnimerNet模型文件，解决了"Error no file named pytorch_model.bin"错误。

## 修复内容

1. 将原始目录中的`pytorch_model.pth`重命名为标准的`pytorch_model.bin`
2. 确保所有必要的配置文件存在

## 原始模型来源

{model_path}

## 修复时间

{import_module('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
    
    logging.info("模型修复完成！")
    logging.info(f"建议更新配置，使用新路径: {output_dir}")
    logging.info("你可以运行以下命令更新配置:")
    logging.info(f"python -m pdf2word.formula.update_model_paths --recognizer={output_dir}")
    
    return output_dir

def import_module(name):
    """动态导入模块"""
    return __import__(name, fromlist=[''])

def main():
    """主函数，处理命令行参数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="修复UnimerNet模型格式问题")
    parser.add_argument("--model", "-m", help="模型路径 (默认为常见下载位置)", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", "-o", help="输出目录 (可选)")
    
    args = parser.parse_args()
    
    fixed_dir = fix_unimernet_model(args.model, args.output)
    
    if fixed_dir:
        print(f"\n修复成功！建议使用以下命令设置新路径:")
        print(f"python -m pdf2word.formula.update_model_paths --recognizer={fixed_dir}")
    else:
        print("\n修复失败，请检查日志信息。")
        sys.exit(1)

if __name__ == "__main__":
    main() 