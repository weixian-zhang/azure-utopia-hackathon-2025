import os
from pathlib import Path

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.csv_loader import CSVLoader


class VectorStore():

    def __init__(self):
        pass
    azure_endpoint: str = os.environ.get('AZURE_OPENAI_ENDPOINT')
    azure_openai_api_key: str = os.environ.get('AZURE_OPENAI_API_KEY')
    azure_openai_api_version: str = "2023-05-15"
    azure_deployment: str = "text-embedding-ada-002"

    vector_store_address: str = os.environ.get('AZURE_SEARCH_ENDPOINT')
    vector_store_password: str = os.environ.get('AZURE_SEARCH_ADMIN_KEY')

    def vectorize_data():
        csv_path = os.path.join(os.path.dirname(__file__), "/rag_data")
        loader = DirectoryLoader(
                path=csv_path,
                glob="**/*.csv",  # This pattern matches all CSV files in the directory and its subdirectories
                loader_cls=CSVLoader(
                    csv_args={
                        "delimiter": ",",

                    },
                )
            )

        happiness_data = loader.load()