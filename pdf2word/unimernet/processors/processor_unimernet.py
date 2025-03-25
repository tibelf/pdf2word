"""
UniMERNet处理器实现
"""

import numpy as np
from PIL import Image
import torch
from transformers import ProcessorMixin

class UniMERNetProcessor(ProcessorMixin):
    """
    UniMERNet预处理器，用于处理输入图像
    """
    
    def __init__(self, image_processor):
        """
        初始化预处理器
        
        Args:
            image_processor: 图像处理器，通常是CLIPImageProcessor
        """
        self.image_processor = image_processor
        
    def __call__(self, images, return_tensors=None, **kwargs):
        """
        处理图像
        
        Args:
            images: 输入图像，可以是PIL Image、numpy数组或已有的张量
            return_tensors: 返回的张量类型，通常是"pt"表示PyTorch张量
            **kwargs: 其他参数
            
        Returns:
            处理后的特征字典
        """
        # 确保输入是列表形式
        if isinstance(images, (Image.Image, np.ndarray, torch.Tensor)):
            images = [images]
            
        # 处理图像
        if self.image_processor is not None:
            # 使用CLIP图像处理器
            image_features = self.image_processor(images, return_tensors=return_tensors, **kwargs)
        else:
            # 简单的预处理（灰度化并调整大小）
            if isinstance(images[0], Image.Image):
                # 预处理PIL图像
                processed_images = []
                for image in images:
                    # 转为灰度图
                    if image.mode != 'L':
                        image = image.convert('L')
                    # 调整大小到固定尺寸
                    image = image.resize((224, 224))
                    # 转为numpy数组并归一化
                    img_array = np.array(image).astype(np.float32) / 255.0
                    processed_images.append(img_array)
                
                # 转为张量
                if return_tensors == "pt":
                    pixel_values = torch.tensor(np.array(processed_images))
                    # 添加通道维度
                    pixel_values = pixel_values.unsqueeze(1)  # [B, 1, H, W]
                    image_features = {"pixel_values": pixel_values}
                else:
                    image_features = {"pixel_values": np.array(processed_images)[:, None, :, :]}
            else:
                # 简单地返回原始张量
                image_features = {"pixel_values": torch.stack(images) if return_tensors == "pt" else np.array(images)}
        
        return image_features 