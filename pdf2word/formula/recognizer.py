'''Formula recognition module using UnimerNet.'''
import logging
import os
import numpy as np
import cv2
from pathlib import Path
import tempfile
import glob
import shutil

from .config import get_model_path
from .direct_load_model import convert_model_format

class FormulaRecognizer:
    '''Recognizes mathematical formulas from images and converts to LaTeX using UnimerNet.'''
    
    def __init__(self, model_path=None):
        '''Initialize formula recognizer.
        
        Args:
            model_path (str, optional): Path to the UnimerNet model for formula recognition.
                If None, will attempt to use a default model.
        '''
        self.model_path = model_path or self._get_default_model_path()
        self.model = None
        self.processor = None
        self.initialized = False
        self.temp_model_dir = None
        
    def _get_default_model_path(self):
        '''Get the default model path.'''
        # Get path from config
        recognizer_dir = get_model_path('FORMULA_RECOGNIZER')
        
        if not recognizer_dir:
            # Fallback to module relative path
            module_dir = os.path.dirname(os.path.abspath(__file__))
            recognizer_dir = os.path.join(module_dir, 'models', 'unimernet_formula_recognition')
            
        return recognizer_dir
    
    def _find_model_files(self, directory):
        '''Find potential model files in the directory structure.
        
        Args:
            directory (str): Directory to search for model files
            
        Returns:
            dict: Dictionary with info about model files found
        '''
        result = {
            'is_valid': False,
            'files': {},
            'search_path': directory
        }
        
        # 标准模型文件名
        model_file_patterns = [
            "*.bin", "*.safetensors", "*.h5", "*.ckpt.*", "*.msgpack", "*.pth",
            "config.json", "*.json", "tokenizer.json", "vocab.txt"
        ]
        
        all_files = []
        # 搜索直接目录
        for pattern in model_file_patterns:
            all_files.extend(glob.glob(os.path.join(directory, pattern)))
            
        # 如果直接目录没有文件，则检查所有子目录（仅一级）
        if not all_files:
            for subdir in os.listdir(directory):
                subdir_path = os.path.join(directory, subdir)
                if os.path.isdir(subdir_path):
                    for pattern in model_file_patterns:
                        matches = glob.glob(os.path.join(subdir_path, pattern))
                        if matches:
                            all_files.extend(matches)
                            # 如果在子目录中找到了文件，更新搜索路径
                            result['search_path'] = subdir_path
                            break
                    if all_files:
                        break
        
        if all_files:
            result['is_valid'] = True
            # 分类找到的文件
            for file_path in all_files:
                file_name = os.path.basename(file_path)
                result['files'][file_name] = file_path
                
        return result
    
    def initialize(self):
        '''Load the UnimerNet model.'''
        if self.initialized:
            return
            
        try:
            # Import here to avoid dependency issues if not using formula recognition
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Formula recognition model not found at {self.model_path}")
            
            # 检查是否有 pytorch_model.pth 文件 - 这是一个特殊情况
            pth_file = os.path.join(self.model_path, "pytorch_model.pth")
            if os.path.exists(pth_file):
                logging.info(f"检测到非标准模型文件: {pth_file}")
                
                # 使用自定义脚本处理模型格式
                self.temp_model_dir = convert_model_format(self.model_path)
                if self.temp_model_dir:
                    logging.info(f"已转换模型格式，使用临时目录: {self.temp_model_dir}")
                    model_path_to_use = self.temp_model_dir
                else:
                    model_path_to_use = self.model_path
            else:
                # 搜索模型文件
                model_info = self._find_model_files(self.model_path)
                
                if not model_info['is_valid']:
                    raise FileNotFoundError(f"No valid model files found in {self.model_path}")
                    
                logging.info(f"找到模型文件，路径: {model_info['search_path']}")
                model_path_to_use = model_info['search_path']
            
            # 尝试加载模型
            try:
                logging.info(f"尝试从 {model_path_to_use} 加载模型")
                self.processor = AutoProcessor.from_pretrained(model_path_to_use, local_files_only=True)
                self.model = AutoModelForVision2Seq.from_pretrained(model_path_to_use, local_files_only=True)
                logging.info("模型加载成功")
            except Exception as first_error:
                logging.warning(f"无法直接加载模型: {first_error}")
                
                # 如果直接加载失败，再次尝试转换格式
                if self.temp_model_dir is None:  # 避免重复转换
                    logging.info("尝试转换模型格式...")
                    self.temp_model_dir = convert_model_format(self.model_path)
                    if not self.temp_model_dir:
                        raise ValueError(f"无法转换模型格式")
                    
                    # 再次尝试加载
                    try:
                        logging.info(f"尝试从转换后的目录加载模型: {self.temp_model_dir}")
                        self.processor = AutoProcessor.from_pretrained(self.temp_model_dir, local_files_only=True)
                        self.model = AutoModelForVision2Seq.from_pretrained(self.temp_model_dir, local_files_only=True)
                        logging.info("模型加载成功")
                    except Exception as second_error:
                        # 清理临时目录
                        self._cleanup_temp_dir()
                        raise ValueError(f"无法加载模型。错误1: {first_error}, 错误2: {second_error}")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
                
            self.initialized = True
            logging.info(f"公式识别模型加载成功，路径: {model_path_to_use}")
        except ImportError:
            logging.error("缺少必要的库。请安装: pip install torch transformers")
            raise
        except Exception as e:
            logging.error(f"初始化公式识别器时出错: {str(e)}")
            raise
    
    def _cleanup_temp_dir(self):
        '''清理临时目录'''
        if self.temp_model_dir and os.path.exists(self.temp_model_dir):
            try:
                shutil.rmtree(self.temp_model_dir, ignore_errors=True)
                self.temp_model_dir = None
            except:
                pass
            
    def __del__(self):
        '''Cleanup temporary directory if it exists.'''
        self._cleanup_temp_dir()
            
    def recognize(self, formula_image):
        '''Recognize formula from image and convert to LaTeX.
        
        Args:
            formula_image (numpy.ndarray): The formula image as a numpy array.
            
        Returns:
            str: The LaTeX representation of the formula.
        '''
        if not self.initialized:
            self.initialize()
            
        try:
            import torch
            
            # Prepare the image
            if len(formula_image.shape) == 2:  # Grayscale
                formula_image = cv2.cvtColor(formula_image, cv2.COLOR_GRAY2RGB)
            elif formula_image.shape[2] == 4:  # RGBA
                formula_image = cv2.cvtColor(formula_image, cv2.COLOR_RGBA2RGB)
                
            # Process the image
            try:
                # 尝试标准处理方式
                inputs = self.processor(images=formula_image, return_tensors="pt")
                
                # 根据处理器类型，可能需要添加text参数
                if hasattr(self.processor, 'tokenizer') and 'text' not in inputs:
                    # 添加空文本作为起始序列
                    inputs['text'] = self.processor.tokenizer([""], return_tensors="pt", padding=True)
                
                # Move to the same device as model
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                    
                # Generate sequences
                with torch.no_grad():
                    try:
                        # 尝试标准生成方式
                        generated_ids = self.model.generate(
                            **inputs,
                            max_length=300,
                            num_beams=5,
                            early_stopping=True
                        )
                    except TypeError as e:
                        # 如果失败，尝试只使用pixel_values
                        if 'pixel_values' in inputs:
                            logging.info("尝试使用替代生成方法...")
                            generated_ids = self.model.generate(
                                pixel_values=inputs['pixel_values'],
                                max_length=300,
                                num_beams=5,
                                early_stopping=True
                            )
                        else:
                            raise e
                    
                # 尝试不同的解码方式
                try:
                    # 尝试标准解码方式
                    latex = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                except:
                    # 尝试直接使用tokenizer解码
                    if hasattr(self.processor, 'tokenizer'):
                        latex = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    else:
                        raise ValueError("无法解码生成的序列")
            
            except Exception as e:
                # 如果第一种方法失败，尝试使用简化的方法
                logging.warning(f"标准方法失败: {e}, 尝试备用方法...")
                
                # 确保图像尺寸合适(224x224是许多视觉模型的标准输入尺寸)
                image = cv2.resize(formula_image, (224, 224))
                # 归一化到[0,1]
                image = image.astype(np.float32) / 255.0
                # 转换为PyTorch张量，添加批次维度
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
                
                if torch.cuda.is_available():
                    image_tensor = image_tensor.to("cuda")
                    self.model = self.model.to("cuda")
                
                # 尝试直接使用图像张量生成
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        image_tensor,
                        max_length=300,
                        num_beams=5,
                        early_stopping=True
                    )
                
                # 尝试解码
                if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'tokenizer'):
                    latex = self.model.decoder.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                elif hasattr(self.processor, 'tokenizer'):
                    latex = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                else:
                    # 最基本的解码 - 许多模型使用这种基本格式
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # 通用分词器
                    latex = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            return latex
        except Exception as e:
            logging.error(f"Error recognizing formula: {str(e)}")
            return "\\text{{Error: {0}}}".format(str(e).replace('{', '{{').replace('}', '}}'))
            
    def process_formula_regions(self, formula_regions):
        '''Process a list of formula regions to recognize LaTeX for each.
        
        Args:
            formula_regions (list): List of formula regions with 'image' key.
            
        Returns:
            list: Updated formula regions with 'latex' key added.
        '''
        for region in formula_regions:
            latex = self.recognize(region['image'])
            region['latex'] = latex
            
        return formula_regions 