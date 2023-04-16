from llama_index import SimpleDirectoryReader, download_loader
from utils.reader.NotionPageReader import NotionPageReader
import os
from pathlib import Path
from utils.reader.NotionPageReader import NotionPageReader
from utils.reader.UnstructuredReader import UnstructuredReader

from dotenv import load_dotenv
load_dotenv(verbose=True)
os.environ["NOTION_INTEGRATION_TOKEN"] = os.getenv('NOTION_INTEGRATION_TOKEN')

def unstructured_loader(file_path: Path):
    reader = UnstructuredReader()
    documents = reader.load_data(file=file_path)
    return documents

def directory_loader(file_path: Path):
    documents = SimpleDirectoryReader(file_path).load_data()
    return documents

def korean_pdf_loader(file_path: Path):
    CJKPDFReader = download_loader("CJKPDFReader")
    loader = CJKPDFReader()
    documents = loader.load_data(file=file_path)
    return documents

def notion_loader(page_ids: list):
    integration_token = os.getenv('NOTION_INTEGRATION_TOKEN')
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