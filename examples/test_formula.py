#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test script for mathematical formula recognition in PDF files.

This script tests the new mathematical formula recognition feature in pdf2word.
"""

import os
import argparse
import logging
from pdf2word.converter import Converter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_pdf(pdf_path, output_path=None, debug=False):
    """Process a PDF file and extract formulas.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_path (str, optional): Path to save the DOCX file. If None, uses the PDF name with .docx extension.
        debug (bool, optional): Enable debug mode. Defaults to False.
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file does not exist: {pdf_path}")
        return
    
    # Set output path if not provided
    if not output_path:
        output_path = os.path.splitext(pdf_path)[0] + '.docx'
    
    # Show options
    logger.info(f"Processing PDF: {pdf_path}")
    logger.info(f"Output DOCX: {output_path}")
    
    try:
        # Create converter
        converter = Converter(pdf_path)
        
        # Set options with focus on formula processing
        options = converter.default_settings
        options['debug'] = debug
        options['process_formulas'] = True
        
        # Convert PDF to DOCX with formula recognition
        converter.convert(output_path, **options)
        
        logger.info(f"Conversion completed successfully. Output saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
    finally:
        # Clean up
        if 'converter' in locals():
            converter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test mathematical formula recognition in PDF files')
    parser.add_argument('pdf_path', help='Path to the PDF file containing formulas')
    parser.add_argument('--output', help='Output DOCX file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    process_pdf(args.pdf_path, args.output, args.debug) 