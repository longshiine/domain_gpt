from llama_index import SimpleDirectoryReader, download_loader
from utils.NotionPageReader import NotionPageReader
import os
from pathlib import Path


os.environ["NOTION_INTEGRATION_TOKEN"] = 'secret_HiVnOZQbGjMZvsMSmBniDKElJYgCLz0Iumtv5026Lj6'

def directory_loader(file_path: Path):
    documents = SimpleDirectoryReader(file_path).load_data()
    return documents

def korean_pdf_loader(file_path: Path):
    CJKPDFReader = download_loader("CJKPDFReader")
    loader = CJKPDFReader()
    documents = loader.load_data(file=file_path)
    return documents

def notion_loader(page_ids: list):
    integration_token = 'secret_HiVnOZQbGjMZvsMSmBniDKElJYgCLz0Iumtv5026Lj6'
    reader = NotionPageReader(integration_token=integration_token)
    documents = reader.load_data(page_ids=page_ids)
    return documents

def web_loader():
    UnstructuredURLLoader = download_loader("UnstructuredURLLoader")
    urls = [
        "https://www.law.go.kr/%EB%B2%95%EB%A0%B9/%EB%AF%BC%EB%B2%95",
    ]
    loader = UnstructuredURLLoader(urls=urls, continue_on_failure=False, headers={"User-Agent": "value"})
    documents = loader.load()
    print(documents)
    return documents