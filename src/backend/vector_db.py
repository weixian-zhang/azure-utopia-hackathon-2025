import os
from pathlib import Path

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.csv_loader import CSVLoader


class VectorStore():

    def __init__(self):
        embedding_model: str = "text-embedding-ada-002"
        vector_store_address: str = os.environ.get('AZURE_SEARCH_ENDPOINT')
        vector_store_password: str = os.environ.get('AZURE_SEARCH_ADMIN_KEY')

        self.embeddings = AzureOpenAIEmbeddings(
            model=embedding_model
        )

        self.vector_store: AzureSearch = AzureSearch(
            azure_search_endpoint=vector_store_address,
            azure_search_key=vector_store_password,
            index_name='happiness_country_index',
            embedding_function=self.embeddings.embed_query
        )

    def load_and_vectorize_csv(self):
        csv_path = os.path.join(os.path.dirname(__file__), "data", "stage_2_happiness_index")
        loader = DirectoryLoader(
                path=csv_path,
                glob="**/*.csv",  # This pattern matches all CSV files in the directory and its subdirectories
                loader_cls=CSVLoader,
                loader_kwargs={
                    "csv_args": {
                        "delimiter": ",",
                    }
                }
            )

        documents = loader.load()


        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        document_ids =self.vector_store.add_documents(docs)

        assert len(document_ids) == len(docs), "RAG documents were not added to the Azure AI Search"


    def similarity_search(self, query: str, k: int = 3):
        result = self.vector_store.similarity_search(query, k=k)
        return result
    

    def has_vector(self) -> bool:
        if not self.similarity_search("Singapore", k=1):
            return False
        return True


    