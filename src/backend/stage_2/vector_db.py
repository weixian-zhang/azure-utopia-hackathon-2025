import os
from pathlib import Path

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
# from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader, Docx2txtLoader

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
            index_name='utopia_info',
            embedding_function=self.embeddings.embed_query
        )

    def load_and_vectorize_csv(self):

        data_path = os.path.join(os.path.dirname(__file__))

        # loader = DirectoryLoader(
        #         path=data_path,
        #         glob="**/*.{pdf,docx}",
        #         loader_kwargs={
        #             "csv_args": {
        #                 "delimiter": ",",
        #             }
        #         },
        #         loader_cls={
        #             ".pdf": PyPDFLoader,
        #             ".docx": Docx2txtLoader
        #         },
        #         recursive=True
        #     )
        
        # documents = loader.load()

        pdf_loader = DirectoryLoader(
            path=data_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            recursive=False,
            show_progress=True
        )
        
        # Load DOCX files
        docx_loader = DirectoryLoader(
            path=data_path,
            glob="**/*.docx",
            loader_cls=Docx2txtLoader,
            recursive=True,
            show_progress=True
        )
        
        # Load both file types
        pdf_documents = pdf_loader.load()
        docx_documents = docx_loader.load()

        documents = pdf_documents + docx_documents


        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        document_ids =self.vector_store.add_documents(docs)

        assert len(document_ids) == len(docs), "RAG documents were not added to the Azure AI Search"


    def similarity_search(self, query: str, k: int = 5):
        result = self.vector_store.similarity_search(query, k=k)
        return result
    

    def has_vector(self) -> bool:
        if not self.similarity_search("Singapore", k=1):
            return False
        return True


    