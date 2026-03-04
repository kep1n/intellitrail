import os
from pathlib import Path
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.rag.loader import load_pdf_documents, load_md_documents
from langchain_text_splitters import MarkdownHeaderTextSplitter
from src.config import Settings
from langchain_pinecone import PineconeVectorStore


def build_index(pdf_dir: str | Path) -> PineconeVectorStore:
    """Load PDFs from pdf_dir, split into chunks, embed, and save FAISS index to index_path.

    Embedding model: text-embedding-3-small (cost-efficient, sufficient for safety docs).
    Chunk size: 800 tokens, overlap: 100 tokens (captures full gear recommendation context).

    Raises ValueError if pdf_dir contains no PDFs — caller must provide documents.

    Saves index to Pinecone.
    """
    settings = Settings()
    os.environ["PINECONE_API_KEY"] = settings.pinecone_api_key
    docs = load_pdf_documents(pdf_dir)

    if not docs:
        raise ValueError(
            f"No PDF documents found in '{pdf_dir}'. "
            "Add alpine safety PDFs to the directory before building the index."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.openai_api_key,
    )

    index_name = create_index_name('mountain-safety-agent')
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
    )

    # Split by markdown headers to preserve structure
    to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3")
    ]

    #ordered_lists = [(f'{i}.', 'Ordered list {i}') for i in range(20)]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=to_split_on, strip_headers=False)
    md_docs = load_md_documents(pdf_dir)
    for md_doc in md_docs:
        chunks = splitter.split_text(md_doc.page_content)
        vectorstore.add_documents(documents=chunks)

    return vectorstore


def create_index_name(index_name: str, dimension: int = 1536, metric: str = 'cosine') -> str:
    """
    Create an index name on Pinecone
    """
    settings = Settings()

    cloud = 'aws'
    region = 'us-east-1'
    pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
    spec = ServerlessSpec(cloud=cloud, region=region)
    if index_name not in pinecone_client.list_indexes().names():
        # initialize the index, and insure the stats are all zeros
        pinecone_client.create_index(name=index_name, dimension=dimension, metric=metric, spec=spec)
    return index_name


def load_index(index_name: str = "mountain-safety-agent") -> PineconeVectorStore:
    """Connect to an existing Pinecone index by name.

    Raises ValueError if the index does not exist on Pinecone.
    Run build_index() first to create and populate it.
    """
    settings = Settings()
    pinecone_client = Pinecone(api_key=settings.pinecone_api_key)

    if index_name not in pinecone_client.list_indexes().names():
        raise ValueError(
            f"Pinecone index '{index_name}' not found. "
            "Run build_index() first to create and populate it."
        )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.openai_api_key,
    )
    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=settings.pinecone_api_key,
    )
