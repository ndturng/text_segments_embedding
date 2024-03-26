from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter

# Read the PDF file from: data/test.pdf
pdf_file = "data/test.pdf"

# Create a PdfReader object
pdf_content = PdfReader(pdf_file)

text = ""
for page in pdf_content.pages:
    text += page.extract_text()

# Split the text into chunks
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
)
text_chunks = splitter.split_text(text)


# Load the SBERT model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)  # There are many other models available

# Embed the text chunks
embeddings = model.encode(text_chunks)
print("Done")
