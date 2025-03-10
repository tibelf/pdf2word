"""Formula converter for recognizing formulas and converting them to LaTeX.

Uses MinerU's MFR (Math Formula Recognition) model to convert formula images to LaTeX.
This module can handle formula recognition with or without the required dependencies.
"""

import os
import logging
import tempfile
from PIL import Image
import numpy as np
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Try to import PyTorch and related dependencies
try:
    import torch
    import onnxruntime
    from transformers import PreTrainedTokenizerFast

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    logger.warning("Formula recognition dependencies not found. Formula recognition will be disabled.")


class FormulaConverter:
    """Class to convert formula images to LaTeX."""
    
    def __init__(self, model_dir=None):
        """Initialize formula converter.
        
        Args:
            model_dir (str, optional): Directory with MFR model files.
                If None, will try to use default location.
        """
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
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
                'MFR',
                'unimernet_small_2501'
            )
        
        # Only attempt to load the model if dependencies are available
        if HAS_DEPS and os.path.exists(model_dir):
            try:
                self._load_model(model_dir)
                self.model_loaded = True
                logger.info(f"Formula recognition model loaded from {model_dir}")
            except Exception as e:
                logger.warning(f"Failed to load formula recognition model: {e}")
        else:
            if not HAS_DEPS:
                logger.warning("Formula recognition is disabled due to missing dependencies")
            else:
                logger.warning(f"Formula recognition model directory not found: {model_dir}")
    
    def _load_model(self, model_dir):
        """Load the ONNX model and tokenizer."""
        # Check for both PyTorch and ONNX models, prefer ONNX if available
        onnx_path = os.path.join(model_dir, 'unimernet_small_2501.onnx')
        tokenizer_path = os.path.join(model_dir, 'tokenizer.json')
        
        if os.path.exists(onnx_path):
            # Load ONNX model
            self.model = onnxruntime.InferenceSession(
                onnx_path,
                providers=['CPUExecutionProvider']
            )
            logger.info("Loaded ONNX model for formula recognition")
        else:
            logger.warning(f"ONNX model not found at {onnx_path}")
            return
            
        # Load tokenizer
        if os.path.exists(tokenizer_path):
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
            logger.info("Loaded tokenizer for formula recognition")
        else:
            logger.warning(f"Tokenizer not found at {tokenizer_path}")
            self.model = None  # Can't use model without tokenizer
    
    def extract_formula_image(self, page, bbox):
        """Extract formula image from a page.
        
        Args:
            page: A PyMuPDF page object
            bbox: Tuple of (x0, y0, x1, y1) in PDF coordinates
            
        Returns:
            PIL.Image: Image of the formula region
        """
        # Get the pixmap for this region
        mat = fitz.Matrix(3, 3)  # Use higher resolution for better recognition
        clip = fitz.Rect(bbox)
        pix = page.get_pixmap(matrix=mat, clip=clip)
        
        # Convert to PIL image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Convert to RGB mode if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        return img
        
    def image_to_latex(self, img):
        """Convert a formula image to LaTeX.
        
        Args:
            img: PIL.Image of the formula
            
        Returns:
            str: LaTeX representation of the formula
        """
        if not self.model_loaded:
            return ""
            
        try:
            # Resize image
            max_size = 800
            width, height = img.size
            if width > max_size or height > max_size:
                scale = max_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Normalize
            img_tensor = img_array.transpose(2, 0, 1).astype(np.float32) / 255.0
            
            # Add batch dimension
            img_tensor = np.expand_dims(img_tensor, axis=0)
            
            # ONNX inference
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            output = self.model.run([output_name], {input_name: img_tensor})
            
            # Process output
            pred_ids = output[0][0]
            
            # Convert to tokens
            latex = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
            
            return self._clean_latex(latex)
        except Exception as e:
            logger.error(f"Error during formula recognition: {e}")
            return ""
            
    def convert_formula(self, page, bbox):
        """Extract and convert a formula from a page.
        
        Args:
            page: A PyMuPDF page object
            bbox: Tuple of (x0, y0, x1, y1) in PDF coordinates
            
        Returns:
            str: LaTeX representation of the formula
        """
        if not self.model_loaded:
            return ""
            
        try:
            img = self.extract_formula_image(page, bbox)
            return self.image_to_latex(img)
        except Exception as e:
            logger.error(f"Error extracting formula: {e}")
            return ""
            
    def _clean_latex(self, latex):
        """Clean up the LaTeX string."""
        # Add $ $ if not already present (for inline formulas)
        if not latex:
            return ""
            
        latex = latex.strip()
        
        if not latex.startswith('$') and not latex.startswith('\\[') and not latex.startswith('\\begin'):
            latex = f"${latex}$"
            
        return latex 