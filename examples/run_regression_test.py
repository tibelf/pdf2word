#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Regression test script for pdf2word.

This script runs a complete regression test on the pdf2word library,
verifying that all functionality works as expected.
"""

import os
import sys
import tempfile
import argparse
import subprocess
import logging
import shutil
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(command, cwd=None):
    """Run a shell command and return the result."""
    logger.info(f"Running command: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"Command completed successfully")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False, e.stderr


def install_formula_deps(conda_env):
    """Install formula recognition dependencies."""
    logger.info("Installing formula recognition dependencies...")
    
    # Activate the conda environment and install dependencies
    cmd = f"conda run -n {conda_env} python install_formula_deps.py --download-models"
    success, output = run_command(cmd)
    
    if not success:
        logger.warning("Failed to install formula dependencies. Formula recognition may not work.")
    
    return success


def run_basic_conversion_test(pdf_path, conda_env):
    """Run a basic conversion test without formula recognition."""
    logger.info("Running basic conversion test...")
    
    # Create a temporary output file
    with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
        output_path = temp_file.name
    
    try:
        # Run the conversion - use a Python script approach rather than CLI
        cmd = f"conda run -n {conda_env} python -c \"from pdf2word.converter import Converter; cv = Converter('{pdf_path}'); cv.convert('{output_path}'); cv.close()\""
        success, output = run_command(cmd)
        
        # Check if output file exists
        if success and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info("Basic conversion test passed!")
            return True
        else:
            logger.error("Basic conversion test failed: Output file not created or empty")
            return False
    finally:
        # Clean up
        if os.path.exists(output_path):
            os.unlink(output_path)


def run_formula_test(pdf_path, conda_env):
    """Run a test with formula recognition enabled."""
    logger.info("Running formula recognition test...")
    
    # Create a temporary output file
    with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
        output_path = temp_file.name
    
    try:
        # Run the conversion with formula recognition using direct Python code
        cmd = f"""conda run -n {conda_env} python -c "
from pdf2word.converter import Converter
cv = Converter('{pdf_path}')
options = cv.default_settings
options['process_formulas'] = True
cv.convert('{output_path}', **options)
cv.close()
print('Formula test completed')
"
"""
        success, output = run_command(cmd)
        
        # Check if output file exists
        if success and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info("Formula recognition test passed!")
            return True
        else:
            logger.error("Formula recognition test failed: Output file not created or empty")
            return False
    finally:
        # Clean up
        if os.path.exists(output_path):
            os.unlink(output_path)


def check_for_bugs(conda_env):
    """Run tests to check for bugs."""
    logger.info("Running bug checks...")
    
    # Import and sanity checks
    cmd = f"conda run -n {conda_env} python -c 'from pdf2word import Converter; print(\"Import test passed!\")'"
    success, output = run_command(cmd)
    
    if not success:
        logger.error("Import test failed!")
        return False
    
    # Formula module checks
    cmd = f"conda run -n {conda_env} python -c 'from pdf2word.formula import FormulaProcessor; print(\"Formula module import test passed!\")'"
    success, output = run_command(cmd)
    
    if not success:
        logger.warning("Formula module import test failed! Formula recognition might not work.")
    
    return True


def check_numpy_version(conda_env):
    """Check NumPy version and fix if needed."""
    logger.info("Checking NumPy version compatibility...")
    
    cmd = f"conda run -n {conda_env} python -c \"import numpy; print('NumPy version:', numpy.__version__); print('NUMPY_VERSION_OK:' + ('0' if numpy.__version__.startswith('2.') else '1'))\""
    success, output = run_command(cmd)
    
    if not success:
        logger.warning("Failed to check NumPy version")
        return
    
    # Check if we need to fix NumPy version
    if 'NUMPY_VERSION_OK:0' in output:
        logger.warning("NumPy 2.x detected. This may cause compatibility issues with formula detection.")
        response = input("Would you like to downgrade NumPy to version 1.x? (y/n): ")
        
        if response.lower() == 'y':
            logger.info("Downgrading NumPy...")
            cmd = f"conda run -n {conda_env} python fix_numpy_issue.py --reinstall-deps"
            run_command(cmd)
            logger.info("NumPy version fixed. Please run the tests again.")
            sys.exit(0)
        else:
            logger.warning("Continuing with NumPy 2.x. Formula detection might not work correctly.")


def main():
    """Run regression tests."""
    parser = argparse.ArgumentParser(description="Run regression tests for pdf2word")
    parser.add_argument('pdf_path', help='Path to a PDF file for testing')
    parser.add_argument('--conda-env', default='pdf2word', help='Conda environment name (default: pdf2word)')
    parser.add_argument('--skip-numpy-check', action='store_true', help='Skip NumPy version check')
    args = parser.parse_args()
    
    # Check if PDF file exists
    if not os.path.exists(args.pdf_path):
        logger.error(f"PDF file does not exist: {args.pdf_path}")
        sys.exit(1)
    
    logger.info("Starting regression tests for pdf2word")
    logger.info(f"Using conda environment: {args.conda_env}")
    logger.info(f"Testing with PDF file: {args.pdf_path}")
    
    # Record start time
    start_time = time.time()
    
    # Step 0: Check NumPy version compatibility
    if not args.skip_numpy_check:
        check_numpy_version(args.conda_env)
    
    # Step 1: Install formula dependencies
    install_formula_deps(args.conda_env)
    
    # Step 2: Check for bugs
    if not check_for_bugs(args.conda_env):
        logger.error("Bug checks failed! Fix the bugs before continuing.")
        sys.exit(1)
    
    # Step 3: Run basic conversion test
    if not run_basic_conversion_test(args.pdf_path, args.conda_env):
        logger.error("Basic conversion test failed! Fix the issues before continuing.")
        sys.exit(1)
    
    # Step 4: Run formula recognition test
    formula_test_result = run_formula_test(args.pdf_path, args.conda_env)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("REGRESSION TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Basic conversion test: PASSED")
    logger.info(f"Formula recognition test: {'PASSED' if formula_test_result else 'FAILED (but not critical)'}")
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    logger.info("="*60)
    
    # Exit with success
    if formula_test_result:
        logger.info("All tests PASSED!")
        sys.exit(0)
    else:
        logger.warning("Some tests FAILED. See log for details.")
        sys.exit(1)


if __name__ == "__main__":
    main() 