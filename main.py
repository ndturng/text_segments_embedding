from PyPDF2 import PdfReader


# Read the PDF file from: data/test.pdf
pdf_file = "data/test.pdf"

# Create a PdfReader object
pdf_content = PdfReader(pdf_file)

text = ""
for page in pdf_content.pages:
    text += page.extract_text()

print(text)
