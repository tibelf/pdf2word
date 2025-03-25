import os
import fitz
from PIL import Image
from pathlib import Path
import numpy as np
import io
import tempfile
import logging

from .formula_processor import FormulaProcessor
from .latex_to_docx import add_latex_equation_to_paragraph

class FormulaIntegration:
    """
    集成数学公式处理到PDF到Word转换的过程中
    """
    
    def __init__(self):
        """初始化公式集成器"""
        # 初始化公式处理器
        self.formula_processor = FormulaProcessor()
        # 用于存储页面公式信息的字典
        self.page_formulas = {}
    
    def process_pdf_formulas(self, pdf_path):
        """
        处理PDF文件中的所有公式
        
        :param pdf_path: PDF文件路径
        :return: 包含所有页面公式信息的字典
        """
        # 打开PDF文件
        doc = fitz.open(pdf_path)
        
        # 处理每一页
        for page_idx in range(len(doc)):
            page = doc.load_page(page_idx)
            
            # 渲染页面为图像
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("ppm")
            
            # 转换为PIL图像
            pil_img = Image.open(io.BytesIO(img_data))
            
            # 处理页面中的公式
            formulas = self.formula_processor.process_page(pil_img)
            
            # 存储公式信息
            self.page_formulas[page_idx] = {
                'formulas': formulas,
                'page_size': (page.rect.width, page.rect.height)
            }
        
        # 关闭PDF文件
        doc.close()
        
        return self.page_formulas
    
    def integrate_formula_to_docx(self, doc, page_idx, text_block, paragraph):
        """
        集成公式到docx文档中的段落
        
        :param doc: docx文档对象
        :param page_idx: 页码索引
        :param text_block: 文本块对象
        :param paragraph: 段落对象
        :return: 是否集成了公式
        """
        try:
            # 处理页码不存在的情况
            if page_idx not in self.page_formulas:
                return False
            
            # 处理text_block可能不是TextBlock对象的情况
            if not hasattr(text_block, 'bbox'):
                return False
            
            # 获取当前页面的公式信息
            page_formulas = self.page_formulas[page_idx]['formulas']
            if not page_formulas:
                return False
            
            # 获取文本块的边界框
            block_bbox = (text_block.bbox.x0, text_block.bbox.y0, 
                        text_block.bbox.x1, text_block.bbox.y1)
            
            # 检查是否有公式与当前文本块重叠
            for formula in page_formulas:
                formula_bbox = formula['bbox']
                
                # 检查边界框是否重叠
                if self._is_overlapping(block_bbox, formula_bbox):
                    # 获取LaTeX公式
                    latex = formula['latex']
                    
                    # 检查公式类型（行内或行间）
                    is_inline = formula['category_id'] == 13  # 13: inline_equation
                    
                    # 将LaTeX公式添加到段落中
                    add_latex_equation_to_paragraph(paragraph, latex, is_inline)
                    
                    return True
            
            return False
        except Exception as e:
            logging.warning(f"Formula processing failed: {e}")
            return False
    
    @staticmethod
    def _is_overlapping(bbox1, bbox2):
        """
        检查两个边界框是否重叠
        
        :param bbox1: 边界框1 (x0, y0, x1, y1)
        :param bbox2: 边界框2 (x0, y0, x1, y1)
        :return: 是否重叠的布尔值
        """
        try:
            # 检查水平方向是否重叠
            h_overlap = not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2])
            
            # 检查垂直方向是否重叠
            v_overlap = not (bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])
            
            # 两个方向都重叠则边界框重叠
            return h_overlap and v_overlap
        except Exception as e:
            logging.warning(f"Error checking overlapping: {e}")
            return False
    
    def clear_cache(self):
        """
        清除缓存的公式信息
        """
        self.page_formulas.clear() 