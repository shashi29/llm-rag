
# Check for installed modules
try:
    import win32com.client
    win32com_client_installed = True
except ImportError:
    win32com_client_installed = False

try:
    import pypandoc
    pypandoc_installed = True
except ImportError:
    pypandoc_installed = False

try:
    from docx2python import docx2python
    docx2python_installed = True
except ImportError:
    docx2python_installed = False

try:
    import textract
    textract_installed = True
except ImportError:
    textract_installed = False

try:
    import fitz  # PyMuPDF
    pymupdf_installed = True
except ImportError:
    pymupdf_installed = False

try:
    from docx import Document
    python_docx_installed = True
except ImportError:
    python_docx_installed = False

def extract_text_using_pywin32(doc_path):
    """
    Extract text from a DOC file using pywin32 (Windows only).
    """
    if not win32com_client_installed:
        raise ImportError("pywin32 is not installed.")
    
    word = win32com.client.Dispatch("Word.Application")
    doc = word.Documents.Open(doc_path)
    text = doc.Content.Text
    doc.Close(False)
    word.Quit()
    return text

def extract_text_using_pypandoc(doc_path):
    """
    Extract text from a DOC file using pypandoc.
    """
    if not pypandoc_installed:
        raise ImportError("pypandoc is not installed.")
    
    text = pypandoc.convert_file(doc_path, 'plain')
    return text

def extract_text_using_docx2python(doc_path):
    """
    Extract text from a DOCX file using docx2python.
    """
    if not docx2python_installed:
        raise ImportError("docx2python is not installed.")
    
    doc = docx2python(doc_path)
    text = '\n'.join([p.text for p in doc.text_paragraphs])
    return text

def extract_text_using_textract(doc_path):
    """
    Extract text from a DOC file using textract.
    """
    if not textract_installed:
        raise ImportError("textract is not installed.")
    
    text = textract.process(doc_path).decode('utf-8')
    return text

def extract_text_using_pymupdf(doc_path):
    """
    Extract text from a DOC file using PyMuPDF (if it's actually a PDF).
    """
    if not pymupdf_installed:
        raise ImportError("PyMuPDF is not installed.")
    
    doc = fitz.open(doc_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def extract_text_using_python_docx(doc_path):
    """
    Extract text from a DOCX file using python-docx.
    """
    if not python_docx_installed:
        raise ImportError("python-docx is not installed.")
    
    doc = Document(doc_path)
    text = '\n'.join([p.text for p in doc.paragraphs])
    return text
