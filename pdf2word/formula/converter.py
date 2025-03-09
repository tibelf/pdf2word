'''LaTeX to MathML converter for Word compatibility.'''
import logging
import os
import tempfile
import subprocess
import re

class MathMLConverter:
    '''Converts LaTeX formulas to MathML format for Word documents.'''
    
    def __init__(self):
        '''Initialize the MathML converter.'''
        self.initialized = False
        
    def initialize(self):
        '''Initialize the converter.'''
        if self.initialized:
            return
            
        try:
            # Check for latex2mathml package
            import latex2mathml
            self.initialized = True
            logging.info("MathML converter initialized successfully.")
        except ImportError:
            logging.error("Failed to import latex2mathml. Please install with: pip install latex2mathml")
            raise
            
    def convert_latex_to_mathml(self, latex):
        '''Convert LaTeX to MathML.
        
        Args:
            latex (str): LaTeX formula string.
            
        Returns:
            str: MathML representation of the formula.
        '''
        if not self.initialized:
            self.initialize()
            
        try:
            import latex2mathml.converter
            
            # Clean the LaTeX string if needed
            latex = self._clean_latex(latex)
            
            # Convert to MathML
            mathml = latex2mathml.converter.convert(latex)
            return mathml
        except Exception as e:
            logging.error(f"Error converting LaTeX to MathML: {str(e)}")
            return f"<math><merror>Error converting formula: {latex}</merror></math>"
    
    def _clean_latex(self, latex):
        '''Clean LaTeX string for better conversion.
        
        Args:
            latex (str): Original LaTeX string.
            
        Returns:
            str: Cleaned LaTeX string.
        '''
        # Remove any \begin{document} or \end{document} tags
        latex = re.sub(r'\\begin\{document\}|\\end\{document\}', '', latex)
        
        # Ensure math mode for standalone formulas
        if not (latex.startswith('$') or latex.startswith('\\(') or 
                latex.startswith('\\[') or latex.startswith('\\begin{align}') or
                latex.startswith('\\begin{equation}')):
            latex = f"${latex}$"
            
        return latex
        
    def process_formula_regions(self, formula_regions):
        '''Convert LaTeX in formula regions to MathML.
        
        Args:
            formula_regions (list): List of formula regions with 'latex' key.
            
        Returns:
            list: Updated formula regions with 'mathml' key added.
        '''
        for region in formula_regions:
            if 'latex' in region and region['latex']:
                mathml = self.convert_latex_to_mathml(region['latex'])
                region['mathml'] = mathml
            else:
                region['mathml'] = ""
                
        return formula_regions
        
    def get_word_compatible_mathml(self, mathml):
        '''Get Word-compatible OMML (Office MathML) format.
        
        Some Word versions might need specific formats or namespace declarations.
        This method ensures compatibility with MS Word.
        
        Args:
            mathml (str): Standard MathML string.
            
        Returns:
            str: Word-compatible OMML string.
        '''
        # Add Word-specific namespaces if needed
        if '<math' in mathml and 'xmlns' not in mathml:
            mathml = mathml.replace('<math', '<math xmlns="http://www.w3.org/1998/Math/MathML"')
            
        return mathml 