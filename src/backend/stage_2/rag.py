from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from stage_2.vector_db import VectorStore
# from vector_db import VectorStore
from dotenv import load_dotenv
from dotenv import load_dotenv
load_dotenv()

class RAG():
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm = AzureChatOpenAI(
            deployment_name="gpt-4o",
            model="gpt-4o",
            api_version="2024-12-01-preview",
            temperature=0.1
        )

    def vectorize_data_if_not_exists(self):
        if not self.vector_store.has_vector():
            self.vector_store.load_and_vectorize_csv()


    def answer_query(self, query: str) -> str:
        # Step 1: Retrieve relevant documents from the vector store
        relevant_docs = self.vector_store.similarity_search(query, k=5)

        # Step 2: Construct the context for the LLM
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        system_prompt = f"""You are an expert assistant for Azure Utopia settlers.

        CONTEXT DOCUMENTS:
        {context}

        INSTRUCTIONS:
        1. Focus on the facts and principles presented in the CONTEXT DOCUMENTS
        2. Focus on main points in the CONTEXT DOCUMENTS and ignore unimportant details
        3. Use a step-by-step reasoning approach to identify relevant information
        4. After reasoning, formulate a clear answer using step by step answer but output a clear answer without Steps
        5. If no relevant information exists, say so explicitly

        example question:
        What ethical principles guide Stellar Horizons' governance and operations?

        example answer: 
        Correctly identifies transparent governance under the Azure Consensus Protocol (ACP) and mentions equity in general.
        two core principles from the reference—justice-by-design and consent-first data practices—and equity baselines for essential services.'

        example question:
        What is the function of the Terraformer Coordinator agent?

        example answer:
        The Terraformer Coordinator agent governs the swarms of programmable nanites used in terraforming. It ensures atmospheric rebalancing, soil enrichment, water synthesis, and microbial seeding proceed according to plan. Embedded governance tokens allow civic oversight to pause, redirect, or decommission terraforming if ecological risks exceed thresholds.
        It also manages biome scaffolding and the Azure Utopia context.

        Question: {query}

        Let's answer step by step:"""

        user_prompt = f"""Question: {query}
Please provide a clear, accurate answer based solely on the context documents."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        # Step 4: Get the answer from the LLM
        response: AIMessage = self.llm.invoke(messages)
        answer = response.content.strip()

        return answer
    

if __name__ == "__main__":
    rag = RAG()
    rag.vectorize_data_if_not_exists()
    q1 = "What are the first tasks to be completed within 72 hours after landing on Azure Utopia"
    q2 = "What is the expected duration of the revival process after cryo-stasis, and why is it important?"
    q3 = "What is the role of AI Health Agents during the mission?"
    q4 = "What ethical principles guide Stellar Horizons' governance and operations?"
    answer = rag.answer_query(q2)
    print(f"Question: {q2}\nAnswer: {answer}")
    

