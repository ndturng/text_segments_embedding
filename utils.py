from langchain.document_loaders import PyPDFLoader

DATA_PATH = "data"


def load_documents():
    loader = PyPDFLoader("example_data/layout-parser-paper.pdf")
    documents = loader.load()
    return documents
