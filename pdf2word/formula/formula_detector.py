"""Formula detector for locating formula regions in PDF documents.

This module can detect mathematical formulas in PDF pages.
"""

import os
import logging
import sys
import numpy as np
from PIL import Image
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# 检查NumPy版本并提出警告
numpy_version = np.__version__
if numpy_version.startswith('2.'):
    logger.warning(f"NumPy 2.x detected (version {numpy_version}). This may cause compatibility issues with YOLO.")
    logger.warning("Consider downgrading NumPy to version <2.0 for better compatibility.")

# Try to import required dependencies with better error handling
HAS_DEPS = False
NUMPY_VERSION_ERROR = False

try:
    # First check if we can safely import YOLO with current NumPy version
    if numpy_version.startswith('2.'):
        # 尝试设置环境变量，禁用NumPy数组API（可能有助于兼容性）
        os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
        # 将警告转为错误来捕获潜在问题
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                from ultralytics import YOLO
                HAS_DEPS = True
                logger.info("Successfully loaded YOLO with NumPy 2.x")
            except Warning as w:
                if '_ARRAY_API not found' in str(w):
                    NUMPY_VERSION_ERROR = True
                    logger.warning(f"YOLO compatibility issue with NumPy 2.x: {w}")
                    logger.warning("Formula detection will be disabled due to NumPy version compatibility issue.")
    else:
        # NumPy 1.x应该没有兼容性问题
        from ultralytics import YOLO
        HAS_DEPS = True
except ImportError:
    logger.warning("Formula detection dependencies (YOLO) not found. Formula detection will be disabled.")
except Exception as e:
    logger.warning(f"Error loading YOLO: {e}")
    logger.warning("Formula detection will be disabled.")


class FormulaDetector:
    """Class to detect formulas in PDF pages."""
    
    def __init__(self, model_dir=None):
        """Initialize formula detector.
        
        Args:
            model_dir (str, optional): Directory with MFD model.
                If None, will try to use default location.
        """
        self.model = None
        self.model_loaded = False
        
        # If NumPy version error occurred, don't even try to load
        if NUMPY_VERSION_ERROR:
            logger.warning("Skipping formula detection model loading due to NumPy version compatibility issues")
            return
        
        # If no model_dir specified, try to use default location
        if model_dir is None:
            home_dir = os.path.expanduser('~')
            model_dir = os.path.join(
                home_dir, 
                '.cache', 
                'modelscope', 
                'hub', 
                'models',
                'opendatalab',
                'PDF-Extract-Kit-1___0',
                'models',
                'MFD',
                'YOLO'
            )
        
        # Only attempt to load the model if dependencies are available
        if HAS_DEPS:
            try:
                model_path = os.path.join(model_dir, 'yolo_v8_ft.pt')
                if os.path.exists(model_path):
                    self.model = YOLO(model_path)
                    self.model_loaded = True
                    logger.info(f"Formula detection model loaded from {model_path}")
                else:
                    logger.warning(f"Formula detection model not found at {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load formula detection model: {e}")
        else:
            logger.warning("Formula detection is disabled due to missing dependencies")
            
    def render_page(self, page, dpi=300):
        """Render a PDF page to an image.
        
        Args:
            page: A PyMuPDF page object
            dpi (int): DPI for rendering
            
        Returns:
            PIL.Image: Rendered page
        """
        # Calculate zoom factor from DPI
        zoom = dpi / 72  # 72 is the default DPI for PDF
        mat = fitz.Matrix(zoom, zoom)
        
        # Render page to pixmap
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        return img
            
    def detect_formulas(self, page):
        """Detect formulas in a PDF page.
        
        Args:
            page: A PyMuPDF page object
            
        Returns:
            list: List of formula bounding boxes as (x0, y0, x1, y1) in PDF coordinates
        """
        if not self.model_loaded:
            return []
            
        try:
            # Render page to image
            img = self.render_page(page)
            
            # Run detection
            results = self.model.predict(
                img, 
                conf=0.25,  # Confidence threshold
                iou=0.45,   # NMS IoU threshold
                verbose=False
            )[0]
            
            # Get detected bounding boxes
            formula_boxes = []
            
            # Image dimensions
            img_width, img_height = img.size
            
            # Page dimensions
            page_width, page_height = page.rect.width, page.rect.height
            
            # Scale factors to convert from image to PDF coordinates
            scale_x = page_width / img_width
            scale_y = page_height / img_height
            
            # Extract bounding boxes
            if hasattr(results, 'boxes') and len(results.boxes) > 0:
                for box in results.boxes:
                    # Get box coordinates
                    x0, y0, x1, y1 = box.xyxy[0].tolist()
                    
                    # Convert to PDF coordinates
                    pdf_x0 = x0 * scale_x
                    pdf_y0 = y0 * scale_y
                    pdf_x1 = x1 * scale_x
                    pdf_y1 = y1 * scale_y
                    
                    formula_boxes.append((pdf_x0, pdf_y0, pdf_x1, pdf_y1))
            
            return formula_boxes
        except Exception as e:
            logger.error(f"Error during formula detection: {e}")
            return [] 