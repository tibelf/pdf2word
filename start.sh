#!/bin/bash
# PDF to Word converter script using conda environment

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Function to show usage
function show_usage {
    echo "Usage: $0 input.pdf [output.docx] [options]"
    echo ""
    echo "Options:"
    echo "  --start=N       First page to convert (0-based index)"
    echo "  --end=N         Last page to convert (0-based index)"
    echo "  --pages=N,M,... Specific pages to convert (0-based index)"
    echo "  --password=XXX  Password for encrypted PDF"
    echo ""
    echo "Examples:"
    echo "  $0 input.pdf"
    echo "  $0 input.pdf output.docx"
    echo "  $0 input.pdf output.docx --start=0 --end=5"
    echo "  $0 input.pdf output.docx --pages=0,2,4"
    echo "  $0 input.pdf output.docx --password=secret"
    exit 1
}

# Check if input file is provided
if [ "$#" -lt 1 ]; then
    show_usage
fi

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH."
    echo "Please install conda from https://docs.conda.io/projects/conda/en/latest/user-guide/install/"
    exit 1
fi

# Check if the environment exists
ENV_NAME="pdf2word"
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "Creating conda environment '$ENV_NAME'..."
    conda env create -f environment.yml
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create conda environment."
        exit 1
    fi
fi

# Install the package in development mode if not installed
if ! conda run -n "$ENV_NAME" python -c "import pdf2word" &> /dev/null; then
    echo "Installing pdf2word package..."
    conda run -n "$ENV_NAME" pip install -e .
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install pdf2word package."
        exit 1
    fi
fi

# Prepare arguments
INPUT_PDF="$1"
shift

# Check if input file exists
if [ ! -f "$INPUT_PDF" ]; then
    echo "Error: Input file '$INPUT_PDF' does not exist."
    exit 1
fi

# If second parameter doesn't start with --, use it as output file
OUTPUT_DOCX=""
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    OUTPUT_DOCX="$1"
    shift
fi

# Convert PDF to Word using the conda environment
echo "Converting PDF to Word document..."
if [ -z "$OUTPUT_DOCX" ]; then
    conda run -n "$ENV_NAME" python -m pdf2word.main convert "$INPUT_PDF" "$@"
else
    conda run -n "$ENV_NAME" python -m pdf2word.main convert "$INPUT_PDF" "$OUTPUT_DOCX" "$@"
fi

# Check if conversion was successful
if [ $? -eq 0 ]; then
    if [ -z "$OUTPUT_DOCX" ]; then
        # Get default output filename (same as input but with .docx extension)
        OUTPUT_DOCX="${INPUT_PDF%.*}.docx"
    fi
    
    if [ -f "$OUTPUT_DOCX" ]; then
        echo "Conversion successful! Output saved to: $OUTPUT_DOCX"
    else
        echo "Warning: Conversion process completed but output file not found."
    fi
else
    echo "Error: Conversion failed."
    exit 1
fi 