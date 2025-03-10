#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fix NumPy version compatibility issues.

This script helps resolve the NumPy 2.x compatibility issues with YOLO by downgrading NumPy.
"""

import os
import sys
import subprocess
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def fix_numpy_version():
    """Fix NumPy version compatibility issues by downgrading to 1.x."""
    logger.info("Checking NumPy version...")
    
    try:
        import numpy as np
        numpy_version = np.__version__
        
        if numpy_version.startswith('2.'):
            logger.warning(f"NumPy 2.x detected (version {numpy_version}).")
            logger.info("Downgrading NumPy to version 1.26.4...")
            
            # Uninstall current NumPy
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "numpy"])
            
            # Install NumPy 1.x
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.26.4"])
            
            logger.info("NumPy downgraded successfully. Please restart your application.")
            return True
        else:
            logger.info(f"NumPy version {numpy_version} is already compatible. No action needed.")
            return True
    except ImportError:
        logger.error("NumPy is not installed. Please run install_formula_deps.py instead.")
        return False
    except Exception as e:
        logger.error(f"Error fixing NumPy version: {e}")
        return False

def main():
    """Main function to fix NumPy version issues."""
    parser = argparse.ArgumentParser(description="Fix NumPy version compatibility issues")
    parser.add_argument("--reinstall-deps", action="store_true", help="Also reinstall other formula dependencies")
    args = parser.parse_args()
    
    # Fix NumPy version
    success = fix_numpy_version()
    
    # Reinstall dependencies if requested
    if success and args.reinstall_deps:
        logger.info("Reinstalling formula dependencies...")
        try:
            from install_formula_deps import install_formula_deps
            install_formula_deps(use_numpy_2=False)
            logger.info("Dependencies reinstalled successfully.")
        except Exception as e:
            logger.error(f"Error reinstalling dependencies: {e}")
    
    logger.info("""
=================================================================
NumPy Version Fix Complete
=================================================================
If you continue to experience issues, please try the following:

1. Run the full dependency installation again:
   python install_formula_deps.py --download-models

2. Ensure you're using a compatible environment with NumPy 1.x

3. If problems persist, you may need to temporarily disable
   formula recognition by setting process_formulas=False
=================================================================
""")

if __name__ == "__main__":
    main() 