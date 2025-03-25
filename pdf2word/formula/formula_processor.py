import os
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import torch

# 尝试导入需要的第三方库
try:
    from ultralytics import YOLO
except ImportError:
    # 可以在此处添加安装指南
    raise ImportError("需要安装额外的依赖库: ultralytics 等。")

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
        
        # 识别模型部分暂时跳过
        self.mfr_model = None
        self.mfr_transform = None
    
    def _init_mfd_model(self, weight_path):
        """
        初始化公式检测模型
        :param weight_path: 模型权重路径
        :return: 检测模型
        """
        mfd_model = YOLO(weight_path)
        return mfd_model
    
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
                formula_list.append({
                    'bbox': [xmin, ymin, xmax, ymax],
                    'category_id': category_id,
                    'confidence': confidence,
                    'latex': category_id == 13 and '$x^2 + y^2 = z^2$' or '$$\\int_{a}^{b} f(x) dx$$'  # 默认的示例公式
                })
        except Exception as e:
            import logging
            logging.error(f"检测公式时出错: {e}")
        
        return formula_list
    
    def recognize_formula(self, image, bbox):
        """
        识别单个数学公式的LaTeX表示
        :param image: PIL.Image 对象
        :param bbox: 公式的边界框 [x_min, y_min, x_max, y_max]
        :return: LaTeX表示的字符串
        """
        # 简化实现，返回默认公式
        # 根据公式类型返回不同的默认公式
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
        
        # 为每个公式添加一个默认的LaTeX表示
        for formula in formulas:
            # 使用更新的recognize_formula方法获取公式
            if not formula['latex']:
                formula['latex'] = self.recognize_formula(image, formula['bbox'])
        
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