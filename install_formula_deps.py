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


def install_formula_deps(cuda=False):
    """Install all dependencies for formula recognition."""
    # Base dependencies
    packages = [
        "onnxruntime" if not cuda else "onnxruntime-gpu",
        "transformers==4.30.2",
        "Pillow>=9.0.0",
        "numpy>=1.20.0",
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
    success = install_formula_deps(cuda=use_cuda)
    
    # Download models if requested
    if args.download_models and success:
        download_models()


if __name__ == "__main__":
    main() 