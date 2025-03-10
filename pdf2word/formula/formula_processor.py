"""Formula processor integrating detection and recognition.

This module provides a high-level interface for detecting and recognizing 
mathematical formulas in PDF documents.
"""

import logging
import os
from .formula_detector import FormulaDetector
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
        self.detector = FormulaDetector(model_dir)
        self.converter = FormulaConverter(model_dir)
        
        # Check if both components are loaded
        self.enabled = self.detector.model_loaded and self.converter.model_loaded
        
        if self.enabled:
            logger.info("Formula processor initialized successfully")
        else:
            logger.warning("Formula processor partially or fully disabled")
            
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
            # Detect formulas
            formula_boxes = self.detector.detect_formulas(page)
            
            # Recognize formulas
            formulas = []
            for bbox in formula_boxes:
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