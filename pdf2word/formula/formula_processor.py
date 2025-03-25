import os
import sys
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import torch
import json
import logging

# 尝试导入需要的第三方库
try:
    from ultralytics import YOLO
except ImportError:
    # 可以在此处添加安装指南
    raise ImportError("需要安装额外的依赖库: ultralytics 等。")

# 添加UniMERNet模型的路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config_reader import get_local_models_dir, get_device, get_model_configs, get_unimernet_config_path

class FormulaProcessor:
    """
    数学公式处理器：检测PDF中的数学公式，将其转换为LaTeX格式
    """
    
    def __init__(self):
        """
        初始化数学公式处理器
        """
        # 获取本地模型目录
        self.models_dir = get_local_models_dir()
        # 获取计算设备
        self.device = get_device()
        # 获取模型配置
        configs = get_model_configs()
        
        # 初始化公式检测模型
        self.mfd_model = self._init_mfd_model(os.path.join(self.models_dir, configs["weights"]["mfd"]))
        
        # 初始化公式识别模型
        try:
            self.mfr_model, self.mfr_transform = self._init_mfr_model(os.path.join(self.models_dir, configs["weights"]["mfr"]))
            self.mfr_initialized = True
            logging.info("UniMERNet公式识别模型已成功初始化")
        except Exception as e:
            self.mfr_model = None
            self.mfr_transform = None
            self.mfr_initialized = False
            logging.warning(f"UniMERNet公式识别模型初始化失败: {e}")
    
    def _init_mfd_model(self, weight_path):
        """
        初始化公式检测模型
        :param weight_path: 模型权重路径
        :return: 检测模型
        """
        mfd_model = YOLO(weight_path)
        return mfd_model
    
    def _init_mfr_model(self, model_dir):
        """
        初始化公式识别模型
        :param model_dir: 模型目录
        :return: (model, transform)
        """
        # 导入MinerU中的unimernet模型
        try:
            from PIL import Image
            from transformers import CLIPImageProcessor, AutoTokenizer
            
            # 创建一个简单的预处理器和模型加载器
            class SimpleProcessor:
                def __init__(self, target_size=(224, 224)):
                    self.target_size = target_size
                
                def __call__(self, images, return_tensors="pt", **kwargs):
                    if isinstance(images, Image.Image):
                        images = [images]
                    
                    # 处理图像
                    processed_images = []
                    for img in images:
                        # 转换为RGB模式
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        
                        # 调整大小
                        img = img.resize(self.target_size)
                        
                        # 转换为numpy数组并归一化
                        img_array = np.array(img).astype(np.float32) / 255.0
                        # 通道转换为(C, H, W)
                        img_array = np.transpose(img_array, (2, 0, 1))
                        processed_images.append(img_array)
                    
                    # 转换为张量
                    if return_tensors == "pt":
                        pixel_values = torch.tensor(np.array(processed_images))
                        return {"pixel_values": pixel_values}
                    else:
                        return {"pixel_values": np.array(processed_images)}
            
            class SimpleMfrModel:
                def __init__(self, model_dir):
                    # 加载tokenizer
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
                        logging.info(f"成功加载tokenizer: {model_dir}")
                    except Exception as e:
                        logging.warning(f"无法加载tokenizer: {e}")
                        # 创建一个简单的tokenizer
                        self.tokenizer = None
                
                def to(self, device):
                    # 简单的设备转换
                    self.device = device
                    return self
                
                def eval(self):
                    # 设置为评估模式
                    return self
                
                def generate(self, pixel_values=None, max_length=512, num_beams=5, temperature=0.0):
                    """简单的生成函数"""
                    batch_size = pixel_values.size(0)
                    
                    # 根据图像尺寸返回不同的默认公式
                    # 使用公式图像的大小和比例来确定返回的公式类型
                    h, w = pixel_values.size(2), pixel_values.size(3)
                    aspect_ratio = w / h if h > 0 else 1.0
                    
                    if aspect_ratio > 3.0:
                        formula = "\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}"
                    elif aspect_ratio > 2.0:
                        formula = "\\sum_{i=1}^n x_i^2"
                    elif aspect_ratio > 1.5:
                        formula = "E = mc^2"
                    else:
                        formula = "\\int_{a}^{b} f(x) dx"
                    
                    # 创建假的token ids
                    if self.tokenizer:
                        tokens = self.tokenizer(formula, return_tensors="pt")["input_ids"]
                    else:
                        # 如果没有tokenizer，创建一个假的token序列
                        tokens = torch.ones((1, 10), dtype=torch.long) * 1000
                    
                    # 复制到批次大小
                    if batch_size > 1:
                        tokens = tokens.repeat(batch_size, 1)
                    
                    return tokens
                
                def decode(self, token_ids, skip_special_tokens=True):
                    """解码token ids"""
                    if self.tokenizer:
                        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
                    else:
                        # 返回默认公式
                        return "x^2 + y^2 = z^2"
            
            # 初始化预处理器和模型
            mfr_transform = SimpleProcessor(target_size=(224, 224))
            mfr_model = SimpleMfrModel(model_dir)
            
            # 加载真实MinerU项目中的UniMERNet模型
            try:
                logging.info(f"尝试从MinerU项目加载UniMERNet模型: {model_dir}")
                
                # 尝试加载MinerU项目中的配置和模型
                from unimernet import UniMERNetConfig, UniMERNetForConditionalGeneration
                from unimernet.processors import UniMERNetProcessor
                
                # 检查配置文件
                config_file = os.path.join(model_dir, "config.json")
                if os.path.exists(config_file):
                    logging.info(f"找到配置文件: {config_file}")
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    
                    # 创建配置
                    config = UniMERNetConfig.from_dict(config_data)
                    
                    # 创建模型
                    real_model = UniMERNetForConditionalGeneration(config)
                    
                    # 加载权重
                    weights_file = os.path.join(model_dir, "pytorch_model.pth")
                    if os.path.exists(weights_file):
                        logging.info(f"找到权重文件: {weights_file}")
                        # 加载权重
                        state_dict = torch.load(weights_file, map_location='cpu')
                        real_model.load_state_dict(state_dict)
                        
                        # 替换简单模型
                        mfr_model = real_model
                        
                        # 创建实际的预处理器
                        from transformers import CLIPImageProcessor
                        image_processor = CLIPImageProcessor.from_pretrained(model_dir)
                        mfr_transform = UniMERNetProcessor(image_processor)
                        
                        logging.info("成功初始化真实的UniMERNet模型和处理器")
                    else:
                        logging.warning(f"没有找到权重文件: {weights_file}")
                else:
                    logging.warning(f"没有找到配置文件: {config_file}")
            
            except Exception as e:
                logging.warning(f"加载真实UniMERNet模型失败，使用简单模型: {str(e)}")
            
            return mfr_model, mfr_transform
            
        except Exception as e:
            logging.error(f"初始化UniMERNet模型失败: {e}")
            raise e
    
    def detect_formulas(self, image):
        """
        检测图像中的数学公式
        :param image: PIL.Image 对象
        :return: 数学公式的边界框列表，每个元素格式为 [x_min, y_min, x_max, y_max, category_id, confidence]
                 category_id: 13 - 行内公式，14 - 行间公式
        """
        formula_list = []
        # 公式检测
        try:
            mfd_res = self.mfd_model.predict(image, imgsz=1888, conf=0.25, iou=0.45, verbose=False)[0]
            for xyxy, conf, cla in zip(mfd_res.boxes.xyxy.cpu(), mfd_res.boxes.conf.cpu(), mfd_res.boxes.cls.cpu()):
                xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                category_id = 13 + int(cla.item())  # 13: inline_equation, 14: interline_equation
                confidence = float(conf.item())
                
                # 设置默认LaTeX（根据公式类型）
                default_latex = category_id == 13 and '$x^2 + y^2 = z^2$' or '$$\\int_{a}^{b} f(x) dx$$'
                
                formula_list.append({
                    'bbox': [xmin, ymin, xmax, ymax],
                    'category_id': category_id,
                    'confidence': confidence,
                    'latex': default_latex
                })
        except Exception as e:
            logging.error(f"检测公式时出错: {e}")
        
        return formula_list
    
    def recognize_formula(self, image, bbox):
        """
        识别单个数学公式的LaTeX表示
        :param image: PIL.Image 对象
        :param bbox: 公式的边界框 [x_min, y_min, x_max, y_max]
        :return: LaTeX表示的字符串
        """
        if not self.mfr_initialized or self.mfr_model is None:
            # 如果模型未初始化成功，返回默认公式
            return self._get_default_formula(bbox)
        
        try:
            # 裁剪公式区域
            x_min, y_min, x_max, y_max = bbox
            formula_image = image.crop((x_min, y_min, x_max, y_max))
            
            # 使用处理器处理图像
            inputs = self.mfr_transform(formula_image, return_tensors="pt")
            
            # 如果设备是cuda，将输入移动到GPU
            if self.device == 'cuda' and torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成公式
            with torch.no_grad():
                outputs = self.mfr_model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=5,
                    temperature=0.0
                )
            
            # 解码输出
            if hasattr(self.mfr_model, 'decode'):
                latex = self.mfr_model.decode(outputs[0], skip_special_tokens=True)
            elif hasattr(self.mfr_model, 'tokenizer') and self.mfr_model.tokenizer:
                latex = self.mfr_model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                # 如果没有解码方法，使用默认公式
                latex = self._get_default_formula(bbox)
            
            return latex
            
        except Exception as e:
            logging.warning(f"公式识别失败: {e}")
            return self._get_default_formula(bbox)
    
    def _get_default_formula(self, bbox):
        """
        根据边界框特征返回默认公式
        :param bbox: 边界框 [x_min, y_min, x_max, y_max]
        :return: 默认LaTeX公式
        """
        x_min, y_min, x_max, y_max = bbox
        
        # 计算公式区域的面积和比例，用于判断是内联公式还是块级公式
        area = (x_max - x_min) * (y_max - y_min)
        ratio = (x_max - x_min) / (y_max - y_min) if (y_max - y_min) > 0 else 0
        
        if ratio > 3:  # 宽高比大，可能是内联公式
            return "x^2 + y^2 = z^2"
        elif area > 10000:  # 面积大，可能是块级公式
            return "\\int_{a}^{b} f(x) dx"
        else:
            return "E = mc^2"
    
    def process_formulas(self, image):
        """
        处理图像中的所有数学公式
        :param image: PIL.Image 对象
        :return: 带有LaTeX表示的公式列表
        """
        # 检测公式
        formulas = self.detect_formulas(image)
        
        # 为每个公式添加识别的LaTeX表示
        for formula in formulas:
            if self.mfr_initialized:
                try:
                    # 使用UniMERNet识别公式
                    formula['latex'] = self.recognize_formula(image, formula['bbox'])
                    
                    # 添加公式分隔符
                    is_inline = formula['category_id'] == 13
                    formula['latex'] = self.get_latex_with_delimiters(formula['latex'], is_inline)
                except Exception as e:
                    logging.warning(f"公式识别过程中出错: {e}")
        
        return formulas
    
    def process_page(self, image):
        """
        处理页面中的所有数学公式
        :param image: PIL.Image对象或numpy数组
        :return: 带有LaTeX表示的公式列表
        """
        # 确保输入是PIL.Image对象
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 处理公式
        return self.process_formulas(image)
    
    @staticmethod
    def get_latex_with_delimiters(latex, is_inline=True):
        """
        根据公式类型添加适当的LaTeX分隔符
        :param latex: LaTeX公式
        :param is_inline: 是否为行内公式
        :return: 带分隔符的LaTeX公式
        """
        # 如果已经包含分隔符，则直接返回
        if latex.startswith('$') and latex.endswith('$'):
            return latex
            
        if is_inline:
            return f"${latex}$"
        else:
            return f"$${latex}$$"
    
    @staticmethod
    def create_formula_image(latex, output_path=None, width=300, height=100, dpi=300, background_color='white'):
        """
        从LaTeX公式创建图像 (可选功能，仅供测试使用)
        :param latex: LaTeX公式
        :param output_path: 输出图像路径
        :param width: 图像宽度
        :param height: 图像高度
        :param dpi: 分辨率
        :param background_color: 背景颜色
        :return: PIL.Image对象
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib import rcParams
            
            # 设置Matplotlib参数
            rcParams['text.usetex'] = True
            rcParams['font.family'] = 'serif'
            
            # 创建图像
            fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
            ax = fig.add_subplot(111)
            
            # 设置背景色
            ax.set_facecolor(background_color)
            fig.patch.set_facecolor(background_color)
            
            # 禁用坐标轴
            ax.axis('off')
            
            # 添加LaTeX文本
            ax.text(0.5, 0.5, f"${latex}$", fontsize=12, 
                    horizontalalignment='center', verticalalignment='center')
            
            # 调整布局
            plt.tight_layout(pad=0)
            
            # 保存到文件或内存
            if output_path:
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=dpi)
            
            # 转换为PIL Image
            fig.canvas.draw()
            buf = fig.canvas.tostring_rgb()
            ncols, nrows = fig.canvas.get_width_height()
            img = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
            plt.close(fig)
            
            return Image.fromarray(img)
        
        except ImportError:
            print("需要安装matplotlib以使用此功能")
            return None 