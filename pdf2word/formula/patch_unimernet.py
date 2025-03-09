#!/usr/bin/env python
'''UnimerNet模型补丁脚本。

此脚本用于处理UnimerNet模型的特殊需求，主要解决两个问题：
1. 模型文件格式问题（.pth vs .bin）
2. 缺少必要的配置文件
3. 处理器配置问题

当模型加载或推理时遇到问题，可以运行此脚本进行修复。
'''

import os
import sys
import json
import shutil
import logging
import tempfile
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# 默认模型路径
DEFAULT_MODEL_PATH = os.path.expanduser("~/.cache/modelscope/hub/models/opendatalab/PDF-Extract-Kit-1___0/models/MFR/unimernet_small_2501")

def create_processor_config(config_dir):
    """创建处理器配置文件
    
    Args:
        config_dir (str): 配置文件目录
        
    Returns:
        bool: 操作是否成功
    """
    processor_config_path = os.path.join(config_dir, "preprocessor_config.json")
    
    # 如果已存在，不创建
    if os.path.exists(processor_config_path):
        with open(processor_config_path, 'r') as f:
            existing_config = json.load(f)
            
        # 检查是否需要添加text_config
        if "text_config" not in existing_config:
            logging.info("更新现有的preprocessor_config.json文件...")
            existing_config["text_config"] = {
                "do_lower_case": False,
                "model_max_length": 512,
                "padding_side": "right",
                "tokenizer_class": "PreTrainedTokenizer"
            }
            with open(processor_config_path, 'w') as f:
                json.dump(existing_config, f, indent=2)
            logging.info("已更新处理器配置文件")
        return True
    
    # 否则创建新的配置文件
    try:
        # 标准ViT预处理器配置
        config = {
            "crop_size": {
                "height": 224,
                "width": 224
            },
            "do_center_crop": True,
            "do_normalize": True,
            "do_resize": True,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "processor_class": "VisionTextDualEncoderProcessor",
            "size": {
                "height": 224,
                "width": 224
            },
            "text_config": {
                "do_lower_case": False,
                "model_max_length": 512,
                "padding_side": "right",
                "tokenizer_class": "PreTrainedTokenizer"
            }
        }
        
        with open(processor_config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logging.info("已创建处理器配置文件")
        return True
    except Exception as e:
        logging.error(f"创建处理器配置文件时出错: {e}")
        return False

def create_model_config(config_dir):
    """创建模型配置文件
    
    Args:
        config_dir (str): 配置文件目录
        
    Returns:
        bool: 操作是否成功
    """
    config_path = os.path.join(config_dir, "config.json")
    
    # 如果已存在，不创建
    if os.path.exists(config_path):
        return True
    
    try:
        # 简化的VisionEncoderDecoderModel配置
        config = {
            "model_type": "vision-encoder-decoder",
            "encoder": {
                "model_type": "vit",
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.0,
                "attention_probs_dropout_prob": 0.0,
                "initializer_range": 0.02,
                "layer_norm_eps": 1e-12,
                "image_size": 224,
                "patch_size": 16,
                "num_channels": 3
            },
            "decoder": {
                "model_type": "gpt2",
                "vocab_size": 50265,
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "attention_probs_dropout_prob": 0.1,
                "max_position_embeddings": 1024,
                "initializer_range": 0.02,
                "layer_norm_eps": 1e-12,
                "pad_token_id": 0,
                "eos_token_id": 2,
                "decoder_start_token_id": 0
            },
            "tokenizer_class": "GPT2Tokenizer",
            "_name_or_path": "unimernet"
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logging.info("已创建模型配置文件")
        return True
    except Exception as e:
        logging.error(f"创建模型配置文件时出错: {e}")
        return False

def create_tokenizer_files(config_dir):
    """创建分词器文件
    
    Args:
        config_dir (str): 配置文件目录
        
    Returns:
        bool: 操作是否成功
    """
    tokenizer_config_path = os.path.join(config_dir, "tokenizer_config.json")
    
    # 如果已存在，不创建
    if os.path.exists(tokenizer_config_path):
        return True
    
    try:
        # 简化的分词器配置
        config = {
            "model_max_length": 512,
            "tokenizer_class": "PreTrainedTokenizer"
        }
        
        with open(tokenizer_config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logging.info("已创建分词器配置文件")
        return True
    except Exception as e:
        logging.error(f"创建分词器配置文件时出错: {e}")
        return False

def patch_model(model_dir, output_dir=None):
    """修补UnimerNet模型，添加必要的配置和支持文件
    
    Args:
        model_dir (str): 模型目录
        output_dir (str, optional): 输出目录，如果不指定则在模型目录旁创建
        
    Returns:
        str: 修补后的模型目录路径
    """
    if not os.path.exists(model_dir):
        logging.error(f"模型目录不存在: {model_dir}")
        return None
    
    # 确定输出目录
    if output_dir is None:
        parent_dir = os.path.dirname(model_dir)
        output_dir = os.path.join(parent_dir, "unimernet_patched")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"将修补后的模型保存到: {output_dir}")
    
    # 复制所有文件
    for item in os.listdir(model_dir):
        source = os.path.join(model_dir, item)
        dest = os.path.join(output_dir, item)
        
        if os.path.isfile(source):
            shutil.copy2(source, dest)
            logging.info(f"已复制文件: {item}")
            
            # 重命名pytorch_model.pth为pytorch_model.bin（如果存在）
            if item == "pytorch_model.pth":
                bin_dest = os.path.join(output_dir, "pytorch_model.bin")
                shutil.copy2(source, bin_dest)
                logging.info("已将pytorch_model.pth复制为pytorch_model.bin")
    
    # 创建或修改必要的配置文件
    create_processor_config(output_dir)
    create_model_config(output_dir)
    create_tokenizer_files(output_dir)
    
    # 创建README记录修补过程
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"""# 修补后的UnimerNet模型

此目录包含修补后的UnimerNet模型文件，解决了各种兼容性问题。

## 修补内容

1. 将原始目录中的 `pytorch_model.pth` 复制为标准的 `pytorch_model.bin`
2. 创建或更新必要的配置文件：config.json, preprocessor_config.json, tokenizer_config.json
3. 添加特殊处理器参数以解决 "text" 和 "text_target" 错误

## 原始模型来源

{model_dir}

## 修补时间

{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
    
    logging.info("\n模型修补完成！")
    logging.info(f"请更新模型路径配置：")
    logging.info(f"python -m pdf2word.formula.update_model_paths --recognizer={output_dir}")
    
    return output_dir

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="修补UnimerNet模型")
    parser.add_argument("--model", "-m", help="模型目录 (默认为常见下载位置)", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", "-o", help="输出目录 (可选)")
    parser.add_argument("--update-config", "-u", action="store_true", help="修补后自动更新配置")
    
    args = parser.parse_args()
    
    patched_dir = patch_model(args.model, args.output)
    
    if patched_dir:
        print(f"\n模型修补成功！")
        
        # 自动更新配置（如果请求）
        if args.update_config:
            try:
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                from pdf2word.formula.update_model_paths import update_model_paths
                
                update_model_paths(recognizer_path=patched_dir)
                print(f"\n已自动更新配置，模型路径已设置为: {patched_dir}")
            except Exception as e:
                print(f"\n自动更新配置失败: {e}")
                print(f"请手动运行以下命令更新配置:")
                print(f"python -m pdf2word.formula.update_model_paths --recognizer={patched_dir}")
        else:
            print(f"\n请运行以下命令更新配置:")
            print(f"python -m pdf2word.formula.update_model_paths --recognizer={patched_dir}")
    else:
        print("\n模型修补失败，请检查日志。")
        sys.exit(1)

if __name__ == "__main__":
    main() 