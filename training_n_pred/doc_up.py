#pip install python-docx

from docx import Document

def read_docx(file_path,length):
    doc = Document(file_path)
    text = []
    for para in doc.paragraphs:
        if para.text.strip():  # avoid empty lines
            text.append(para.text)

    text_data = " ".join(text)        
    text=text_data[:length]
    return text

# Example usage
#print(read_docx('Conversation.docx',13000))

