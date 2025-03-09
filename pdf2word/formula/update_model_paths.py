#!/usr/bin/env python
'''Update model paths in the configuration file.

This script allows users to update the model paths in the configuration
file to match where their models are actually located.
'''

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

# Now we can import our modules
from pdf2word.formula.config import MODELS

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.py')


def update_model_paths(detector_path=None, recognizer_path=None):
    '''Update the model paths in the config file.
    
    Args:
        detector_path (str, optional): Path to the formula detector model.
        recognizer_path (str, optional): Path to the formula recognizer model.
    '''
    if not detector_path and not recognizer_path:
        print("No paths provided. Nothing to update.")
        return
    
    # Read the current config file
    with open(CONFIG_FILE, 'r') as f:
        lines = f.readlines()
    
    # Update the paths
    updated_lines = []
    for line in lines:
        if detector_path and "'FORMULA_DETECTOR':" in line:
            # Find the line with the detector path and replace it
            indent = line.find("'FORMULA_DETECTOR'")
            updated_lines.append(f"{' ' * indent}'FORMULA_DETECTOR': '{detector_path}',\n")
        elif recognizer_path and "'FORMULA_RECOGNIZER':" in line:
            # Find the line with the recognizer path and replace it
            indent = line.find("'FORMULA_RECOGNIZER'")
            updated_lines.append(f"{' ' * indent}'FORMULA_RECOGNIZER': '{recognizer_path}',\n")
        else:
            updated_lines.append(line)
    
    # Write the updated config back to the file
    with open(CONFIG_FILE, 'w') as f:
        f.writelines(updated_lines)
    
    print("Configuration updated successfully.")
    print(f"Detector model path: {detector_path if detector_path else MODELS['FORMULA_DETECTOR']}")
    print(f"Recognizer model path: {recognizer_path if recognizer_path else MODELS['FORMULA_RECOGNIZER']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Update model paths in the configuration.')
    parser.add_argument('--detector', type=str, help='Path to the formula detector model', default=None)
    parser.add_argument('--recognizer', type=str, help='Path to the formula recognizer model', default=None)
    
    args = parser.parse_args()
    
    update_model_paths(args.detector, args.recognizer)