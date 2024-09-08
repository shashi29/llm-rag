import io
import pdfplumber
import textract
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import numpy as np
# from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import easyocr

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using multiple methods in order.
    """
    methods = [
        extract_text_using_pdfplumber,
        extract_text_using_textract,
        extract_text_using_pdfminer,
        extract_text_using_easyocr,
        # extract_text_using_paddleocr
    ]
    
    for method in methods:
        try:
            text = method(pdf_path)
            if text:
                return text
        except Exception as e:
            print(f"Method {method.__name__} failed: {e}")
    
    raise Exception(f"Failed to extract text from {pdf_path} using all available methods.")

def extract_text_using_pdfplumber(pdf_path):
    """
    Extract text from a PDF file using pdfplumber.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''.join(page.extract_text() for page in pdf.pages)
        return text
    except Exception as e:
        raise Exception(f"pdfplumber extraction failed: {e}")

def extract_text_using_textract(pdf_path):
    """
    Extract text from a PDF file using textract.
    """
    try:
        text = textract.process(pdf_path).decode('utf-8')
        return text
    except Exception as e:
        raise Exception(f"textract extraction failed: {e}")

def extract_text_using_pdfminer(pdf_path):
    """
    Extract text from a PDF file using pdfminer.
    """
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    
    try:
        with open(pdf_path, 'rb') as file_handle:
            for page in PDFPage.get_pages(file_handle, caching=True, check_extractable=True):
                page_interpreter.process_page(page)
            
            text = fake_file_handle.getvalue()
    finally:
        converter.close()
        fake_file_handle.close()
    
    return text


def extract_text_using_easyocr(pdf_path):
    """
    Extract text from a PDF file using easyocr.
    """
    try:
        reader = easyocr.Reader(['en'])
        images = convert_from_path(pdf_path)
        pages_data = []
        for page_number, image in enumerate(images):
            image = np.array(image)
            result = reader.readtext(image, detail=0)
            text = ' '.join(result)
            pages_data.append({
                'page_number': page_number + 1,
                'text': text
            })
        return '\n'.join(page['text'] for page in pages_data)
    except Exception as e:
        raise Exception(f"easyocr extraction failed: {e}")

# def extract_text_using_paddleocr(pdf_path):
#     """
#     Extract text from a PDF file using PaddleOCR.
#     """
#     try:
#         ocr = PaddleOCR()
#         result = ocr.ocr(pdf_path)
#         text = '\n'.join([line for page in result for line in page[1]])
#         return text
#     except Exception as e:
#         raise Exception(f"PaddleOCR extraction failed: {e}")
