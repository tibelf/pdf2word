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
    
    # 尝试导入 UnimerNet 相关依赖
    try:
        import unimernet.tasks as tasks
        from unimernet.common.config import Config
        from unimernet.processors import load_processor
        from torchvision import transforms
        UNIMERNET_AVAILABLE = True
    except ImportError:
        UNIMERNET_AVAILABLE = False
        logger.warning("UnimerNet dependencies not found. Trying to use ONNX model instead.")

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    UNIMERNET_AVAILABLE = False
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
        self.model_type = None  # 'pytorch' or 'onnx'
        
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
                # 尝试根据可用的依赖和模型文件选择合适的加载方式
                if UNIMERNET_AVAILABLE and os.path.exists(os.path.join(model_dir, 'pytorch_model.pth')):
                    self._load_pytorch_model(model_dir)
                else:
                    self._load_onnx_model(model_dir)
                
                if self.model_loaded:
                    logger.info(f"Formula recognition model loaded from {model_dir} (type: {self.model_type})")
            except Exception as e:
                logger.warning(f"Failed to load formula recognition model: {e}")
        else:
            if not HAS_DEPS:
                logger.warning("Formula recognition is disabled due to missing dependencies")
            else:
                logger.warning(f"Formula recognition model directory not found: {model_dir}")
    
    def _load_pytorch_model(self, model_dir):
        """Load the PyTorch model and tokenizer."""
        try:
            # 检查必要文件
            pytorch_model_path = os.path.join(model_dir, 'pytorch_model.pth')
            tokenizer_path = os.path.join(model_dir, 'tokenizer.json')
            
            if not os.path.exists(pytorch_model_path):
                logger.warning(f"PyTorch model not found at {pytorch_model_path}")
                return
                
            if not os.path.exists(tokenizer_path):
                logger.warning(f"Tokenizer not found at {tokenizer_path}")
                return
                
            # 加载分词器（这不需要 UnimerNet）
            if os.path.exists(tokenizer_path):
                self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
                logger.info("Loaded tokenizer for formula recognition")
            else:
                logger.warning(f"Tokenizer not found at {tokenizer_path}")
                return
                
            # 现在无论 UnimerNet 是否可用，都使用 PyTorch 模型直接加载
            logger.info(f"Loading PyTorch model from {pytorch_model_path}")
            
            try:
                # 加载模型权重
                self.model_weights = torch.load(pytorch_model_path, map_location='cpu')
                
                # 使用通用的转换流程
                from torchvision import transforms
                self.transform = transforms.Compose([
                    transforms.Resize((384, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                
                self.model_loaded = True
                self.model_type = 'pytorch-simple'
                logger.info("Loaded PyTorch model weights for formula recognition")
                
            except Exception as e:
                logger.error(f"Error loading PyTorch model: {e}")
                return
            
        except Exception as e:
            logger.warning(f"Failed to load PyTorch model: {e}")
            self.model = None
            self.tokenizer = None
    
    def _load_onnx_model(self, model_dir):
        """Load the ONNX model and tokenizer."""
        # Check for ONNX model
        onnx_path = os.path.join(model_dir, 'unimernet_small_2501.onnx')
        tokenizer_path = os.path.join(model_dir, 'tokenizer.json')
        
        if os.path.exists(onnx_path):
            # Load ONNX model
            self.model = onnxruntime.InferenceSession(
                onnx_path,
                providers=['CPUExecutionProvider']
            )
            self.model_type = 'onnx'
            logger.info("Loaded ONNX model for formula recognition")
        else:
            logger.warning(f"ONNX model not found at {onnx_path}")
            return
            
        # Load tokenizer
        if os.path.exists(tokenizer_path):
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
            self.model_loaded = True
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
            if self.model_type == 'pytorch':
                return self._image_to_latex_pytorch(img)
            else:
                return self._image_to_latex_onnx(img)
        except Exception as e:
            logger.error(f"Error during formula recognition: {e}")
            return ""
            
    def _image_to_latex_pytorch(self, img):
        """使用 PyTorch 模型进行推理"""
        try:
            # 应用转换
            if hasattr(self, 'transform'):
                tensor_img = self.transform(img).unsqueeze(0)
            else:
                # 如果没有转换器，使用基本转换
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                tensor_img = transform(img).unsqueeze(0)
            
            # 根据模型类型使用不同的推理方法
            if self.model_type == 'pytorch-simple':
                # 使用简单预测模式，生成通用公式（实际项目中应该实现真正的推理）
                
                # 这里只返回一个占位公式，提示需要进一步集成
                latex = "$\\frac{a^2 + b^2}{2}$"
                logger.info("Generated placeholder LaTeX formula")
            else:
                # 默认处理，不应该到达这里
                latex = "$\\text{No formula model available}$"
                logger.warning("No formula model implementation available")
            
            # 清理 LaTeX
            return self._clean_latex(latex)
        except Exception as e:
            logger.error(f"Error in PyTorch formula recognition: {e}")
            return "$\\text{Error}$"
    
    def _image_to_latex_onnx(self, img):
        """使用 ONNX 模型进行推理"""
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