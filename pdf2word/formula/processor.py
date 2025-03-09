'''Formula processing pipeline coordinator.'''
import logging
import cv2
import numpy as np
import fitz
from pathlib import Path
import os

from .detector import FormulaDetector
from .recognizer import FormulaRecognizer
from .converter import MathMLConverter

class FormulaProcessor:
    '''Coordinates the entire formula processing pipeline.'''
    
    def __init__(self, detector_model=None, recognizer_model=None):
        '''Initialize the formula processor.
        
        Args:
            detector_model (str, optional): Path to formula detection model.
            recognizer_model (str, optional): Path to formula recognition model.
        '''
        self.detector = FormulaDetector(model_path=detector_model)
        self.recognizer = FormulaRecognizer(model_path=recognizer_model)
        self.converter = MathMLConverter()
        self.initialized = False
        
    def initialize(self):
        '''Initialize all components in the pipeline.'''
        if self.initialized:
            return
            
        try:
            # Initialize each component
            self.detector.initialize()
            self.recognizer.initialize()
            self.converter.initialize()
            
            self.initialized = True
            logging.info("Formula processor initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing formula processor: {str(e)}")
            raise
            
    def process_page(self, page_obj, dpi=300):
        '''Process a PDF page to detect and recognize formulas.
        
        Args:
            page_obj (fitz.Page): PyMuPDF page object.
            dpi (int, optional): DPI for rendering page image. Defaults to 300.
            
        Returns:
            list: List of processed formula regions with bbox, latex, and mathml.
        '''
        if not self.initialized:
            self.initialize()
            
        try:
            # Render page to image
            pix = page_obj.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            
            # Convert to RGB if needed
            if pix.n == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                
            # Step 1: Detect formula regions
            detections = self.detector.detect(img)
            formula_regions = self.detector.extract_formula_regions(img, detections)
            
            # Step 2: Recognize LaTeX from formula regions
            formula_regions = self.recognizer.process_formula_regions(formula_regions)
            
            # Step 3: Convert LaTeX to MathML
            formula_regions = self.converter.process_formula_regions(formula_regions)
            
            # Convert coordinates from image space to PDF space
            scale = 72 / dpi
            for region in formula_regions:
                x1, y1, x2, y2 = region['bbox']
                region['pdf_bbox'] = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
                
            return formula_regions
        except Exception as e:
            logging.error(f"Error processing page for formulas: {str(e)}")
            return []
            
    def process_document(self, doc, start=0, end=None, pages=None):
        '''Process a PDF document to detect and recognize formulas on all specified pages.
        
        Args:
            doc (fitz.Document): PyMuPDF document object.
            start (int, optional): First page to process. Defaults to 0.
            end (int, optional): Last page to process. Defaults to None (last page).
            pages (list, optional): List of specific pages to process. Defaults to None.
            
        Returns:
            dict: Dictionary mapping page numbers to lists of formula regions.
        '''
        if not self.initialized:
            self.initialize()
            
        # Determine pages to process
        if pages is not None:
            page_indices = pages
        else:
            if end is None:
                end = doc.page_count - 1
            page_indices = range(start, end + 1)
            
        # Process each page
        results = {}
        for page_idx in page_indices:
            if 0 <= page_idx < doc.page_count:
                page = doc[page_idx]
                formula_regions = self.process_page(page)
                results[page_idx] = formula_regions
                logging.info(f"Processed page {page_idx+1}: Found {len(formula_regions)} formulas")
            else:
                logging.warning(f"Page index {page_idx} out of range, skipping")
                
        return results 