import os
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from PIL import Image

class FormulaExporter:
    """
    导出检测到的公式及其LaTeX表示的工具类
    """
    
    def __init__(self, export_dir=None):
        """
        初始化导出器
        
        :param export_dir: 导出目录，默认为pdf2word/formula_exports
        """
        # 设置导出目录
        if export_dir:
            self.export_dir = export_dir
        else:
            # 使用默认目录
            self.export_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'formula_exports')
        
        # 确保导出目录存在
        os.makedirs(self.export_dir, exist_ok=True)
        
        # 为当前任务创建一个时间戳子目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_export_dir = os.path.join(self.export_dir, f"export_{timestamp}")
        os.makedirs(self.current_export_dir, exist_ok=True)
        
        # 创建图像目录
        self.images_dir = os.path.join(self.current_export_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        
        # 初始化页面计数器
        self.page_counter = 0
        
        # 存储所有公式的数据
        self.formulas_data = {
            "timestamp": timestamp,
            "pages": {}
        }
        
        logging.info(f"公式导出目录已创建: {self.current_export_dir}")
    
    def export_page_formulas(self, page_idx, page_image, formulas):
        """
        导出一个页面的所有公式
        
        :param page_idx: 页面索引
        :param page_image: 页面图像 (PIL.Image)
        :param formulas: 检测到的公式列表
        """
        # 创建页面目录
        page_dir = os.path.join(self.current_export_dir, f"page_{page_idx}")
        os.makedirs(page_dir, exist_ok=True)
        
        # 保存页面图像
        page_image_path = os.path.join(self.images_dir, f"page_{page_idx}.png")
        page_image.save(page_image_path)
        
        # 收集页面公式数据
        page_data = {
            "page_index": page_idx,
            "page_image": os.path.relpath(page_image_path, self.current_export_dir),
            "formula_count": len(formulas),
            "formulas": []
        }
        
        # 处理每个公式
        for i, formula in enumerate(formulas):
            formula_idx = i + 1
            bbox = formula["bbox"]
            category_id = formula["category_id"]
            latex = formula["latex"]
            is_inline = category_id == 13  # 13: inline_equation
            
            # 裁剪公式图像
            try:
                formula_image = page_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                formula_image_path = os.path.join(page_dir, f"formula_{formula_idx}.png")
                formula_image.save(formula_image_path)
                
                # 收集公式数据
                formula_data = {
                    "formula_index": formula_idx,
                    "bbox": bbox,
                    "category": "inline_equation" if is_inline else "interline_equation",
                    "latex": latex,
                    "image": os.path.relpath(formula_image_path, self.current_export_dir)
                }
                
                page_data["formulas"].append(formula_data)
                
            except Exception as e:
                logging.error(f"导出公式图像时出错: {e}")
        
        # 将页面数据添加到整体数据中
        self.formulas_data["pages"][str(page_idx)] = page_data
        
        # 更新页面计数器
        self.page_counter += 1
        
        # 导出当前页面的JSON数据
        with open(os.path.join(page_dir, "formulas.json"), "w", encoding="utf-8") as f:
            json.dump(page_data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"页面 {page_idx} 的公式已导出: {len(formulas)} 个公式")
    
    def export_summary(self, pdf_path=None):
        """
        导出所有公式的摘要信息
        
        :param pdf_path: PDF文件路径，用于记录
        :return: 导出目录的路径
        """
        # 添加PDF路径信息（如果有）
        if pdf_path:
            self.formulas_data["pdf_path"] = pdf_path
        
        self.formulas_data["total_pages"] = self.page_counter
        
        # 计算公式总数
        total_formulas = sum(len(page_data["formulas"]) for page_data in self.formulas_data["pages"].values())
        self.formulas_data["total_formulas"] = total_formulas
        
        # 导出完整的JSON数据
        with open(os.path.join(self.current_export_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(self.formulas_data, f, ensure_ascii=False, indent=2)
        
        # 创建一个简单的HTML报告
        self._generate_html_report()
        
        logging.info(f"公式导出摘要已生成: 共 {self.page_counter} 页，{total_formulas} 个公式")
        logging.info(f"导出目录: {self.current_export_dir}")
        
        return self.current_export_dir
    
    def _generate_html_report(self):
        """
        生成HTML格式的报告
        """
        html_path = os.path.join(self.current_export_dir, "report.html")
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>数学公式导出报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        .formula {{ border: 1px solid #ddd; margin: 10px 0; padding: 10px; border-radius: 5px; }}
        .formula img {{ max-width: 300px; border: 1px solid #eee; }}
        .latex {{ font-family: monospace; background-color: #f5f5f5; padding: 5px; border-radius: 3px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>数学公式导出报告</h1>
    <p>导出时间: {self.formulas_data.get('timestamp')}</p>
"""
        
        if "pdf_path" in self.formulas_data:
            html_content += f"    <p>PDF文件: {self.formulas_data['pdf_path']}</p>\n"
        
        html_content += f"""    <p>总页数: {self.formulas_data.get('total_pages', 0)}</p>
    <p>总公式数: {self.formulas_data.get('total_formulas', 0)}</p>
    
    <h2>页面概览</h2>
    <table>
        <tr>
            <th>页码</th>
            <th>公式数量</th>
            <th>查看详情</th>
        </tr>
"""
        
        # 添加页面表格
        for page_idx, page_data in sorted(self.formulas_data["pages"].items(), key=lambda x: int(x[0])):
            page_number = int(page_idx) + 1  # 页码从1开始
            formula_count = page_data["formula_count"]
            html_content += f"""        <tr>
            <td>{page_number}</td>
            <td>{formula_count}</td>
            <td><a href="#page_{page_idx}">查看详情</a></td>
        </tr>
"""
        
        html_content += "    </table>\n\n"
        
        # 添加每个页面的详细内容
        for page_idx, page_data in sorted(self.formulas_data["pages"].items(), key=lambda x: int(x[0])):
            page_number = int(page_idx) + 1  # 页码从1开始
            formula_count = page_data["formula_count"]
            page_image = page_data["page_image"]
            
            html_content += f"""    <h2 id="page_{page_idx}">页面 {page_number}</h2>
    <p>公式数量: {formula_count}</p>
    <p><img src="{page_image}" alt="页面 {page_number}" style="max-width: 200px;"></p>
    
    <h3>公式列表</h3>
"""
            
            # 添加每个公式的详细信息
            for formula in page_data["formulas"]:
                formula_idx = formula["formula_index"]
                category = formula["category"]
                latex = formula["latex"]
                image_path = formula["image"]
                
                html_content += f"""    <div class="formula">
        <h4>公式 {formula_idx}</h4>
        <p>类型: {category}</p>
        <p>LaTeX: <span class="latex">{latex}</span></p>
        <p><img src="{image_path}" alt="公式 {formula_idx}"></p>
    </div>
"""
            
            html_content += "\n"
        
        html_content += """</body>
</html>"""
        
        # 写入HTML文件
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logging.info(f"HTML报告已生成: {html_path}")

# 单例模式
_formula_exporter_instance = None

def get_formula_exporter(export_dir=None):
    """
    获取FormulaExporter的单例实例
    
    :param export_dir: 导出目录，默认为None
    :return: FormulaExporter实例
    """
    global _formula_exporter_instance
    if _formula_exporter_instance is None:
        _formula_exporter_instance = FormulaExporter(export_dir)
    return _formula_exporter_instance 