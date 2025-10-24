from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from vector_db import VectorStore
from dotenv import load_dotenv

load_dotenv()

class RAG():
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm = AzureChatOpenAI(
            deployment_name="gpt-4o",
            model="gpt-4o",
            api_version="2024-12-01-preview",
            temperature=1.0
        )

    def vectorize_data_if_not_exists(self):
        if not self.vector_store.has_vector():
            self.vector_store.load_and_vectorize_csv()


    def answer_query(self, query: str) -> str:
        # Step 1: Retrieve relevant documents from the vector store
        relevant_docs = self.vector_store.similarity_search(query, k=3)

        # Step 2: Construct the context for the LLM
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

        Context:
        {context}

        Question: {query}

        Instructions:
        - Answer based ONLY on the information provided in the context above
        - If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question."
        - Be specific and cite relevant details from the context
        - Keep your answer concise and relevant

        Answer:"""

        # Step 4: Get the answer from the LLM
        response: AIMessage = self.llm.invoke([prompt])
        answer = response.content.strip()

        return answer