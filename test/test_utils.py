from utils import load_documents


def test_load_documents():
    documents = load_documents()
    assert len(documents) == 1
