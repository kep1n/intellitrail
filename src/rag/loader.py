from pathlib import Path
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import PyPDFParser
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


def load_pdf_documents(pdf_dir: str | Path) -> list[Document]:
    """Load all PDFs from pdf_dir into LangChain Documents.

    Uses GenericLoader with PyPDFParser (current LangChain recommended pattern).
    Uses TextParser for *.d and *.txt files copied from other resources

    Each page becomes one Document with metadata {'source': filename, 'page': N}.

    Returns empty list if:
    - pdf_dir does not exist
    - pdf_dir contains no .pdf files

    Does NOT raise on empty directory — the caller (build_index) handles this.
    """
    extension = '*.pdf'
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        return []

    pdf_files = list(pdf_dir.glob(extension))
    if not pdf_files:
        return []

    loader = GenericLoader.from_filesystem(
        str(pdf_dir),
        glob=extension,
        parser=PyPDFParser(),
    )
    return loader.load()


def load_md_documents(md_dir):
    extension = '*.md'
    md_directory = Path(md_dir)
    if not md_directory.exists():
        return []
    md_files = list(md_directory.glob(extension))
    if not md_files:
        return []

    for md_file in md_files:
        loader = TextLoader(md_file)
        yield from loader.load()
