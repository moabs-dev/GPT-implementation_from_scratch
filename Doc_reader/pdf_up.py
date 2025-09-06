import fitz  # PyMuPDF (pip install pymupdf)

def read_pdf(file_path,length):
    """Extract text from PDF file using PyMuPDF."""
    doc = fitz.open(file_path)
    text = []
    for page in doc:
        page_text = page.get_text("text")
        if page_text.strip():
            text.append(page_text)
    text_data= " ".join(text)
    text=text_data[:length]
    return text

#print(read_pdf('dc machine book.pdf',1002))