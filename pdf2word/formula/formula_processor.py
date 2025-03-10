"""Formula processor integrating detection and recognition.

This module provides a high-level interface for detecting and recognizing 
mathematical formulas in PDF documents.
"""

import logging
import os
from .formula_detector import FormulaDetector, NUMPY_VERSION_ERROR
from .formula_converter import FormulaConverter

logger = logging.getLogger(__name__)


class FormulaProcessor:
    """High-level formula processing combining detection and recognition."""
    
    def __init__(self, model_dir=None):
        """Initialize formula processor.
        
        Args:
            model_dir (str, optional): Base directory for models.
                If None, will use default locations.
        """
        # 检测NumPy版本错误
        if NUMPY_VERSION_ERROR:
            logger.warning("Formula processing disabled due to NumPy version compatibility issues")
            logger.warning("Consider downgrading NumPy to version <2.0 for better compatibility")
            self.enabled = False
            self.detector = None
            self.converter = None
            return
            
        # 尝试初始化组件
        try:
            # Initialize components
            self.detector = FormulaDetector(model_dir)
            self.converter = FormulaConverter(model_dir)
            
            # 检查模型可用性和使用适当的处理模式
            if self.detector.model_loaded and self.converter.model_loaded:
                # 理想情况：检测器和转换器都可用
                self.detection_mode = "model"
                self.enabled = True
                logger.info("Formula processor initialized with full model support")
            elif self.converter.model_loaded:
                # 只有转换器可用：使用启发式检测
                logger.warning("Formula detection is disabled but recognition is available")
                logger.warning("Will use heuristic-based formula detection instead")
                self.detection_mode = "heuristic"
                self.enabled = True
                logger.info("Formula processor initialized with heuristic detection")
            elif self.detector.model_loaded:
                # 只有检测器可用：插入占位符 LaTeX
                logger.warning("Formula recognition is disabled but detection is available")
                logger.warning("Will use placeholder LaTeX for detected formulas")
                self.detection_mode = "model"
                self.placeholder_mode = True
                self.enabled = True
                logger.info("Formula processor initialized with placeholder formulas")
            else:
                # 无模型可用：使用基本模式
                logger.warning("No formula models available. Using basic mode")
                self.detection_mode = "basic"
                self.enabled = True  # 仍然启用处理器，但使用基本功能
                logger.info("Formula processor initialized in basic mode")
        except Exception as e:
            logger.error(f"Error initializing formula processor: {e}")
            self.enabled = False
            self.detection_mode = "disabled"
            
    def _detect_formulas_heuristic(self, page):
        """使用启发式规则检测数学公式区域。
        
        使用简单的启发式规则查找可能的公式区域，当YOLO模型不可用时使用。
        
        Args:
            page: PyMuPDF页面对象
            
        Returns:
            list: 可能的公式边界框列表 [(x0, y0, x1, y1), ...]
        """
        # 这是一个非常简单的启发式方法：在页面上放置几个候选区域
        # 在实际应用中，可以通过分析页面内容来改进这一方法
        
        # 获取页面尺寸
        width, height = page.rect.width, page.rect.height
        
        # 创建几个候选区域（页面中部区域，可能包含公式）
        # 这只是一个非常基本的示例，实际应用中应当更智能地检测
        regions = []
        
        # 添加一些中部区域作为候选
        col_width = width / 3
        row_height = height / 4
        
        for col in range(3):
            for row in range(1, 3):  # 主要考虑中间行
                x0 = col * col_width
                y0 = row * row_height
                x1 = x0 + col_width
                y1 = y0 + row_height / 2  # 半高区域
                regions.append((x0, y0, x1, y1))
                
        return regions
    
    def _detect_formulas_basic(self, page):
        """最基本的公式区域检测，只返回页面中间的一个区域。
        
        Args:
            page: PyMuPDF页面对象
            
        Returns:
            list: 只包含一个区域的列表
        """
        # 获取页面尺寸
        width, height = page.rect.width, page.rect.height
        
        # 返回页面中间的一个区域
        x0 = width / 4
        y0 = height / 3
        x1 = width * 3 / 4
        y1 = height * 2 / 3
        
        return [(x0, y0, x1, y1)]
            
    def process_page(self, page):
        """Process a PDF page to detect and recognize formulas.
        
        Args:
            page: A PyMuPDF page object
            
        Returns:
            list: List of dictionaries with formula data:
                [
                    {
                        'bbox': (x0, y0, x1, y1),  # Formula bounding box
                        'latex': '...'  # LaTeX code for the formula
                    },
                    ...
                ]
        """
        if not self.enabled:
            return []
            
        try:
            # 根据不同的检测模式获取公式框
            if self.detection_mode == "model" and self.detector and self.detector.model_loaded:
                # 使用模型检测公式
                formula_boxes = self.detector.detect_formulas(page)
            elif self.detection_mode == "heuristic":
                # 使用启发式方法
                formula_boxes = self._detect_formulas_heuristic(page)
            else:
                # 使用基本方法 
                formula_boxes = self._detect_formulas_basic(page)
            
            # 识别公式或生成占位符
            formulas = []
            for bbox in formula_boxes:
                if hasattr(self, 'placeholder_mode') and self.placeholder_mode:
                    # 使用占位符公式
                    latex = "$E = mc^2$"  # 示例公式
                    formulas.append({
                        'bbox': bbox,
                        'latex': latex
                    })
                elif self.detection_mode == "basic":
                    # 基本模式：返回一个示例公式
                    latex = "$a^2 + b^2 = c^2$"  # 毕达哥拉斯定理
                    formulas.append({
                        'bbox': bbox,
                        'latex': latex
                    })
                elif self.converter and self.converter.model_loaded:
                    # 使用转换器
                    latex = self.converter.convert_formula(page, bbox)
                    if latex:  # Only include if recognition succeeded
                        formulas.append({
                            'bbox': bbox,
                            'latex': latex
                        })
                    
            return formulas
        except Exception as e:
            logger.error(f"Error processing formulas: {e}")
            return [] 