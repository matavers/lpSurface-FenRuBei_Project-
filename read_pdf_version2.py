import PyPDF2
import os

# 读取PDF文件的函数
def read_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
            return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# 测试读取算法version2.pdf文件
print("=== 算法version2.pdf ===")
print(read_pdf("算法version2.pdf"))
