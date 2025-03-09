'''Formula detection module using YOLOv8.'''
import logging
import numpy as np
import cv2
from pathlib import Path
import os
import glob

from .config import get_model_path

class FormulaDetector:
    '''Detects mathematical formulas in PDF pages using YOLOv8.'''
    
    def __init__(self, model_path=None):
        '''Initialize formula detector.
        
        Args:
            model_path (str, optional): Path to the YOLOv8 model for formula detection.
                If None, will attempt to use a default model.
        '''
        self.model_path = model_path or self._get_default_model_path()
        self.model = None
        self.initialized = False
        
    def _get_default_model_path(self):
        '''Get the default model path.'''
        # Get path from config
        detector_dir = get_model_path('FORMULA_DETECTOR')
        
        if not detector_dir:
            # Fallback to module relative path
            module_dir = os.path.dirname(os.path.abspath(__file__))
            detector_dir = os.path.join(module_dir, 'models')
        
        # 尝试查找目录中的模型文件
        if os.path.isdir(detector_dir):
            # 查找.pt或.pth文件
            model_files = glob.glob(os.path.join(detector_dir, "*.pt")) + \
                         glob.glob(os.path.join(detector_dir, "*.pth"))
            
            if model_files:
                # 使用找到的第一个模型文件
                return model_files[0]
            else:
                # 如果目录中没有找到模型文件，尝试寻找子目录中的模型
                for subdir in os.listdir(detector_dir):
                    subdir_path = os.path.join(detector_dir, subdir)
                    if os.path.isdir(subdir_path):
                        model_files = glob.glob(os.path.join(subdir_path, "*.pt")) + \
                                    glob.glob(os.path.join(subdir_path, "*.pth"))
                        if model_files:
                            return model_files[0]
        
        # 如果是文件路径，直接返回
        if os.path.isfile(detector_dir):
            return detector_dir
            
        # 默认路径
        module_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(module_dir, 'models', 'formula_detection_yolov8.pt')
        
    def initialize(self):
        '''Load the YOLOv8 model.'''
        if self.initialized:
            return
            
        try:
            # Import here to avoid dependency issues if not using formula detection
            from ultralytics import YOLO
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Formula detection model not found at {self.model_path}")
                
            self.model = YOLO(self.model_path)
            self.initialized = True
            logging.info(f"Formula detection model loaded successfully from {self.model_path}")
        except ImportError:
            logging.error("Failed to import ultralytics. Please install with: pip install ultralytics")
            raise
        except Exception as e:
            logging.error(f"Error initializing formula detector: {str(e)}")
            raise
    
    def detect(self, page_image):
        '''Detect formulas in a page image.
        
        Args:
            page_image (numpy.ndarray): The page image as a numpy array.
            
        Returns:
            list: List of detected formula regions as [x1, y1, x2, y2, confidence].
        '''
        if not self.initialized:
            self.initialize()
            
        try:
            # Run YOLOv8 detection
            results = self.model(page_image)
            
            # Extract bounding boxes
            detections = []
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confs = result.boxes.conf.cpu().numpy()
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    confidence = confs[i]
                    detections.append([x1, y1, x2, y2, confidence])
            
            return detections
        except Exception as e:
            logging.error(f"Error detecting formulas: {str(e)}")
            return []
    
    def extract_formula_regions(self, page_image, detections, padding=5):
        '''Extract formula regions from the page image.
        
        Args:
            page_image (numpy.ndarray): The page image.
            detections (list): List of formula detections [x1, y1, x2, y2, confidence].
            padding (int, optional): Padding to add around formula regions. Defaults to 5.
            
        Returns:
            list: List of cropped formula images.
        '''
        formula_regions = []
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            
            # Add padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(page_image.shape[1], x2 + padding)
            y2 = min(page_image.shape[0], y2 + padding)
            
            # Crop the region
            formula_region = page_image[y1:y2, x1:x2].copy()
            formula_regions.append({
                'image': formula_region,
                'bbox': [x1, y1, x2, y2]
            })
        
        return formula_regions 