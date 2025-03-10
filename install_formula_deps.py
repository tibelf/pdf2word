#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Helper script to install formula recognition dependencies.
This script handles the installation of dependencies required for formula recognition.
"""

import os
import sys
import subprocess
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_environment():
    """Check if running in conda environment."""
    return "CONDA_PREFIX" in os.environ


def check_gpu():
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except (ImportError, ModuleNotFoundError):
        # If torch isn't installed yet, try to detect NVIDIA GPU
        try:
            result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return result.returncode == 0
        except FileNotFoundError:
            return False


def install_pip_package(package):
    """Install a package using pip."""
    logger.info(f"Installing {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package}: {e}")
        return False


def install_formula_deps(cuda=False, use_numpy_2=False):
    """Install all dependencies for formula recognition.
    
    Args:
        cuda (bool): Whether to install CUDA-enabled packages
        use_numpy_2 (bool): Whether to use NumPy 2.x (may cause compatibility issues)
    """
    # First install NumPy (specific version)
    if use_numpy_2:
        logger.info("Installing NumPy 2.x (note: may cause compatibility issues with YOLO)")
        numpy_package = "numpy>=2.0.0"
    else:
        logger.info("Installing NumPy 1.x for better compatibility")
        numpy_package = "numpy<2.0.0"
    
    install_pip_package(numpy_package)
    
    # Base dependencies
    packages = [
        "onnxruntime" if not cuda else "onnxruntime-gpu",
        "transformers==4.30.2",
        "Pillow>=9.0.0",
        "ultralytics==8.0.196"
    ]
    
    # Install PyTorch with or without CUDA
    if cuda:
        logger.info("Installing PyTorch with CUDA support...")
        torch_cmd = "pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118"
        try:
            subprocess.check_call(torch_cmd, shell=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install PyTorch with CUDA: {e}")
            return False
    else:
        logger.info("Installing PyTorch (CPU version)...")
        torch_cmd = "pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu"
        try:
            subprocess.check_call(torch_cmd, shell=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install PyTorch: {e}")
            return False
            
    # 安装 UnimerNet 所需的其他依赖
    logger.info("Installing UnimerNet dependencies...")
    unimernet_deps = [
        "timm==0.4.12",
        "protobuf<=3.20.0",
        "sentencepiece>=0.1.99"
    ]
    
    for pkg in unimernet_deps:
        install_pip_package(pkg)
    
    # 安装 UnimerNet 包，假设它在 MinerU 项目下
    try:
        import sys
        import os
        
        # 检查 MinerU 项目目录是否存在
        mineru_path = os.path.expanduser("~/Github/MinerU")
        if os.path.exists(mineru_path):
            logger.info(f"Adding MinerU project path to PYTHONPATH: {mineru_path}")
            
            # 创建或更新 .pth 文件在 site-packages 目录中以添加 MinerU 项目路径
            import site
            site_packages = site.getsitepackages()[0]
            pth_file = os.path.join(site_packages, "mineru.pth")
            
            with open(pth_file, "w") as f:
                f.write(mineru_path + "\n")
            
            logger.info(f"Created path file at {pth_file}")
        else:
            logger.warning(f"MinerU project directory not found at {mineru_path}")
    except Exception as e:
        logger.error(f"Failed to set up MinerU path: {e}")
    
    # Install all other dependencies
    success = all(install_pip_package(pkg) for pkg in packages)
    
    if success:
        logger.info("Successfully installed all formula recognition dependencies")
    else:
        logger.warning("Formula recognition may not work properly")
    
    return success


def download_models():
    """Download formula recognition models."""
    logger.info("Downloading formula recognition models...")
    try:
        # Import the download script
        # This assumes the download_models.py script is in the same directory
        from download_models import download_onnx_models
        
        # Download models
        success = download_onnx_models()
        
        if success:
            logger.info("Successfully downloaded formula recognition models")
        else:
            logger.warning("Failed to download some formula recognition models")
        
        return success
    except ImportError:
        logger.error("Could not find download_models.py")
        return False


def main():
    """Main function to install dependencies."""
    parser = argparse.ArgumentParser(description="Install formula recognition dependencies")
    parser.add_argument("--cuda", action="store_true", help="Install with CUDA support")
    parser.add_argument("--download-models", action="store_true", help="Download formula recognition models")
    parser.add_argument("--numpy2", action="store_true", help="Use NumPy 2.x (may cause compatibility issues)")
    args = parser.parse_args()
    
    # Check environment
    if not check_environment():
        logger.warning("Not running in a conda environment. It's recommended to run this in a conda environment.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            logger.info("Installation aborted.")
            return
    
    # Check if CUDA is available
    has_cuda = check_gpu()
    use_cuda = args.cuda and has_cuda
    
    if args.cuda and not has_cuda:
        logger.warning("CUDA was requested but no GPU was detected. Falling back to CPU installation.")
    
    # Install dependencies
    logger.info("Installing formula recognition dependencies...")
    success = install_formula_deps(cuda=use_cuda, use_numpy_2=args.numpy2)
    
    # If NumPy 2.x was requested, show compatibility notice
    if args.numpy2:
        logger.warning("You've chosen to use NumPy 2.x, which may cause compatibility issues with YOLO.")
        logger.warning("If formula detection doesn't work, try reinstalling with NumPy 1.x (without --numpy2 flag).")
    
    # Download models if requested
    if args.download_models and success:
        download_models()


if __name__ == "__main__":
    main() 