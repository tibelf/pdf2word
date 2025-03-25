import os
import platform
from pathlib import Path
import yaml
import pkg_resources

def get_device():
    """
    获取计算设备类型
    :return: str: 'cuda' 或 'cpu'
    """
    # 这里简化为仅返回CPU，实际使用时可以根据是否有GPU来判断
    return 'cpu'

def get_local_models_dir():
    """
    获取本地模型目录
    :return: str: 模型目录路径
    """
    # 使用与pdf2word相同的目录结构
    # 从models目录的相对位置获取模型
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

def get_model_configs():
    """
    读取模型配置
    :return: dict: 模型配置信息
    """
    try:
        # 尝试从包资源中获取配置文件路径
        config_path = pkg_resources.resource_filename('pdf2word', 'resources/model_config/model_configs.yaml')
    except:
        # 如果不是以包的形式安装，则使用相对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(os.path.dirname(current_dir), 'resources', 'model_config', 'model_configs.yaml')
    
    with open(config_path, "r", encoding='utf-8') as f:
        configs = yaml.safe_load(f)
    
    return configs

def get_unimernet_config_path():
    """
    获取UniMERNet配置路径
    :return: str: 配置文件路径
    """
    try:
        # 尝试从包资源中获取配置文件路径
        config_path = pkg_resources.resource_filename('pdf2word', 'resources/model_config/UniMERNet/demo.yaml')
    except:
        # 如果不是以包的形式安装，则使用相对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(os.path.dirname(current_dir), 'resources', 'model_config', 'UniMERNet', 'demo.yaml')
    
    return config_path 