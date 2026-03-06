import PyPDF2
import sys

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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_pdf_generic.py <pdf_file_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    print(f"=== Reading {pdf_path} ===")
    content = read_pdf(pdf_path)
    print(content)
