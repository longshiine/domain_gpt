import re
import fitz
from PyPDF2 import PdfWriter, PdfReader


def pdf2text(path, text_path):
    text = ''
    doc = fitz.open(path)
    for i, page in enumerate(doc):
        if i > 1 and i < 531:
            if i not in range(16, 34):
                sub = page.get_text()
                text += sub

    cleaned_text = re.sub(r' \n', '', text)
    with open(text_path, 'w', encoding='utf-8') as fp:
        fp.write(cleaned_text)

def pdf_crop(path, cropped_path):
    reader = PdfReader(path)
    writer = PdfWriter()

    page = reader.pages[125]
    print(page.cropbox.upper_right)
    for i, page in enumerate(reader.pages):
        page.cropbox.upper_left = (55,60)
        page.cropbox.lower_right = (429,646.25)
        writer.add_page(page)

    with open(cropped_path, 'wb') as fp:
        writer.write(fp)


if __name__ == "__main__":
    path = "../data/book.pdf"
    cropped_path = "../data/book_cropped.pdf"
    text_path = "../data/book_text2.txt"
    
    ### for heuristics ###
    # pdf_crop(path, cropped_path)
    pdf2text(cropped_path, text_path)