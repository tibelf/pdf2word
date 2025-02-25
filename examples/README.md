# pdf2word 使用示例

本目录包含了 pdf2word 的使用示例，帮助您快速了解和测试 pdf2word 的功能。

## 测试脚本

### test_convert.py

这是一个简单的测试脚本，用于测试 PDF 到 Word 的转换功能。

**使用方法：**

```bash
# 基本用法
python test_convert.py your_file.pdf

# 指定输出文件
python test_convert.py your_file.pdf -o output.docx

# 指定页面范围
python test_convert.py your_file.pdf -s 0 -e 5

# 带密码的 PDF
python test_convert.py your_file.pdf -p your_password
```

**参数说明：**

- `pdf_file`：必需参数，要转换的 PDF 文件路径
- `-o, --output`：可选参数，输出的 Word 文件路径（默认为与 PDF 同名的 .docx 文件）
- `-s, --start`：可选参数，起始页（0-based，默认为 0）
- `-e, --end`：可选参数，结束页（0-based，默认为最后一页）
- `-p, --password`：可选参数，PDF 密码（如果有）

## 使用 start.sh 脚本

您也可以使用项目根目录下的 `start.sh` 脚本，它会自动设置 conda 环境并执行转换：

```bash
# 基本用法
../start.sh your_file.pdf output.docx

# 指定页面范围
../start.sh your_file.pdf output.docx --start=0 --end=5

# 带密码的 PDF
../start.sh your_file.pdf output.docx --password=your_password
``` 