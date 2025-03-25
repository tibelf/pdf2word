"""
UniMERNet配置类
"""

from transformers import PretrainedConfig

class UniMERNetConfig(PretrainedConfig):
    """
    UniMERNet模型的配置类，继承自transformers的PretrainedConfig
    """
    model_type = "unimernet"
    
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        max_seq_len=1536,
        **kwargs
    ):
        """
        初始化配置
        
        Args:
            vision_config: 视觉模型的配置
            text_config: 文本模型的配置
            projection_dim: 投影维度
            logit_scale_init_value: 初始logit缩放值
            max_seq_len: 最大序列长度
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        
        # 保存参数
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.max_seq_len = max_seq_len
        
        # 视觉配置
        self.vision_config = vision_config
        if isinstance(self.vision_config, dict):
            self.vision_config = dict(self.vision_config)
        
        # 文本配置
        self.text_config = text_config
        if isinstance(self.text_config, dict):
            self.text_config = dict(self.text_config)
            
    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        从配置字典创建配置对象
        """
        config = cls(**config_dict)
        # 设置属性
        for key, value in kwargs.items():
            setattr(config, key, value)
        return config 