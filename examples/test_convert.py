#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
一个简单的 pdf2word 测试脚本
"""

import os
import sys
import argparse

# 添加项目根目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)

from pdf2word import convert

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试 pdf2word 转换功能')
    parser.add_argument('pdf_file', help='要转换的PDF文件路径')
    parser.add_argument('-o', '--output', help='输出的Word文件路径 (默认为与PDF同名的.docx文件)')
    parser.add_argument('-s', '--start', type=int, default=0, help='起始页 (0-based)')
    parser.add_argument('-e', '--end', type=int, help='结束页 (0-based)')
    parser.add_argument('-p', '--password', help='PDF密码 (如果有)')
    args = parser.parse_args()

    # 检查PDF文件是否存在
    if not os.path.exists(args.pdf_file):
        print(f"错误: PDF文件 '{args.pdf_file}' 不存在")
        return 1

    # 设置输出文件路径
    output_file = args.output
    if not output_file:
        base_name = os.path.splitext(args.pdf_file)[0]
        output_file = f"{base_name}.docx"

    print(f"正在转换 '{args.pdf_file}' 到 '{output_file}'...")
    try:
        # 执行转换
        convert(
            pdf_file=args.pdf_file,
            docx_file=output_file,
            password=args.password,
            start=args.start,
            end=args.end
        )
        print(f"转换成功！输出文件: {output_file}")
        return 0
    except Exception as e:
        print(f"转换失败: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 