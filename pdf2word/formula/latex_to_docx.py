import os
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def add_latex_equation_to_paragraph(paragraph, latex_equation, is_inline=True):
    """
    向段落中添加LaTeX格式的数学公式
    
    :param paragraph: docx段落对象
    :param latex_equation: LaTeX格式的数学公式
    :param is_inline: 是否为行内公式
    :return: None
    """
    # 在Word中，公式使用OMML (Office Math Markup Language)格式
    # 这里我们直接在文档中创建一个公式对象，并设置其属性
    
    if is_inline:
        # 创建行内公式对象
        run = paragraph.add_run()
        r = run._r
        equation_element = OxmlElement('m:oMathPara')
        equation_element.append(OxmlElement('m:oMath'))
        
        # 向omml元素添加LaTeX文本运行
        latex_run = OxmlElement('m:r')
        latex_text = OxmlElement('w:t')
        latex_text.text = latex_equation
        latex_run.append(latex_text)
        
        equation_element.xpath('./m:oMath')[0].append(latex_run)
        r.append(equation_element)
    else:
        # 创建行间公式，居中对齐
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run()
        r = run._r
        equation_element = OxmlElement('m:oMathPara')
        equation_element.append(OxmlElement('m:oMath'))
        
        # 向omml元素添加LaTeX文本运行
        latex_run = OxmlElement('m:r')
        latex_text = OxmlElement('w:t')
        latex_text.text = latex_equation
        latex_run.append(latex_text)
        
        equation_element.xpath('./m:oMath')[0].append(latex_run)
        r.append(equation_element)

def add_latex_as_omml(document, latex_formula, is_inline=True):
    """
    将LaTeX公式作为OMML格式添加到文档中
    
    :param document: docx文档对象
    :param latex_formula: LaTeX格式的公式
    :param is_inline: 是否为行内公式
    :return: 添加了公式的段落对象
    """
    if is_inline:
        paragraph = document.add_paragraph()
        add_latex_equation_to_paragraph(paragraph, latex_formula, is_inline=True)
    else:
        paragraph = document.add_paragraph()
        add_latex_equation_to_paragraph(paragraph, latex_formula, is_inline=False)
    
    return paragraph 