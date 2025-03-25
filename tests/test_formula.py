import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pdf2word
from pdf2word.formula.formula_processor import FormulaProcessor
from pdf2word.formula.formula_integration import FormulaIntegration
from pdf2word.converter import Converter

def test_formula_detection():
    """测试公式检测功能"""
    # 初始化公式处理器
    processor = FormulaProcessor()
    
    # 加载测试PDF文件
    input_pdf = "/Users/tibelf/Github/input.pdf"
    
    # 确保测试PDF文件存在
    if not os.path.exists(input_pdf):
        logging.error(f"测试文件不存在: {input_pdf}")
        return
    
    # 初始化公式集成器
    integration = FormulaIntegration()
    
    # 处理PDF中的公式
    formulas = integration.process_pdf_formulas(input_pdf)
    
    # 打印公式信息
    print(f"检测到的公式总数: {sum(len(page_info['formulas']) for page_info in formulas.values())}")
    for page_idx, page_info in formulas.items():
        page_formulas = page_info['formulas']
        print(f"页面 {page_idx+1} 的公式数量: {len(page_formulas)}")
        for i, formula in enumerate(page_formulas):
            print(f"  公式 {i+1}: {formula['latex']} (边界框: {formula['bbox']})")

def test_convert_with_formulas():
    """测试带公式处理的PDF转Word"""
    # 输入和输出文件
    input_pdf = "/Users/tibelf/Github/input.pdf"
    output_docx = "/Users/tibelf/Github/output_with_formulas.docx"
    
    # 确保测试PDF文件存在
    if not os.path.exists(input_pdf):
        logging.error(f"测试文件不存在: {input_pdf}")
        return
    
    # 使用Converter转换PDF到Word
    converter = Converter(input_pdf)
    converter.convert(output_docx, process_formulas=True)
    
    print(f"转换完成，输出文件: {output_docx}")

def test_export_formulas():
    """测试导出公式信息"""
    # 输入和输出文件
    input_pdf = "/Users/tibelf/Github/input.pdf"
    output_docx = "/Users/tibelf/Github/output_with_formulas_exported.docx"
    
    # 确保测试PDF文件存在
    if not os.path.exists(input_pdf):
        logging.error(f"测试文件不存在: {input_pdf}")
        return
    
    # 使用Converter转换PDF到Word，并启用公式导出
    converter = Converter(input_pdf)
    converter.convert(output_docx, process_formulas=True, export_formulas=True)
    
    # 获取导出目录
    export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pdf2word', 'formula_exports')
    if os.path.exists(export_dir):
        exports = sorted([d for d in os.listdir(export_dir) if d.startswith('export_')], reverse=True)
        if exports:
            latest_export = os.path.join(export_dir, exports[0])
            print(f"公式已导出到: {latest_export}")
            print(f"可以打开 {os.path.join(latest_export, 'report.html')} 查看详细信息")
    
    print(f"转换完成，输出文件: {output_docx}")

if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 测试公式检测
    test_formula_detection()
    
    # 测试带公式处理的PDF转Word
    test_convert_with_formulas()
    
    # 测试导出公式信息
    test_export_formulas() 