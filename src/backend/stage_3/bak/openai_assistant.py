from openai import AzureOpenAI, OpenAI
import os
from typing import List, Tuple, Any
import time
# from stage_3.cosmosdb_mongo_manager import CosmosMongoDBManager

class OpenAIAssistant:
    def __init__(self):

        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
            api_version="2024-12-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment='gpt-4o'
        )
        self.assistant = None
        self.thread = None
        self.vector_store = None
        self.assistant_id = None

        #self.cosmos_mongo = CosmosMongoDBManager()

    def setup_assistant(self, max_tokens: int = 150) -> str:
        
        self.create_vector_store_with_files()

        self.create_assistant_with_file_search()

        self.thread = self.client.beta.threads.create()

    
    def _is_vector_store_exists(self) -> bool:
        """
        Check if vector store exists
        """

        vector_store = None
        vector_stores = self.client.vector_stores.list(limit=1, order="desc")
        for vs in vector_stores.data:
            vector_store = vs

        if vector_store:
            self.vector_store = vector_store
            return True,
        
        return False
    
    def _is_assistant_exist(self) -> bool:
        """
        Check if an assistant exists
        """

        assistant = None
        assistants = self.client.beta.assistants.list(limit=1, order="desc")
        for ap in assistants.data:
            assistant = ap

        if assistant:
            self.assistant = assistant
            return True,
        
        return False
    

    def create_vector_store_with_files(self):
        """
        Create a vector store and upload files to it
        """
        selection_criteria_data_path = os.path.join(os.path.dirname(__file__), "selection_criteria.json")
        vector_store_name = "selection_criteria_vector_store"
        
        # Upload files and create vector store
        file_stream = open(selection_criteria_data_path, "rb")
        file_stream = [file_stream]

        try:

            if self._is_vector_store_exists():
                return self.vector_store.id
            
            # Create vector store with files
            self.vector_store = self.client.vector_stores.create(
                name=vector_store_name,
                file_ids=[]
            )
        
            # Upload files to vector store
            file_batch = self.client.vector_stores.file_batches.upload_and_poll(
                vector_store_id=self.vector_store.id,
                files=file_stream
            )
            
            print(f"Vector store created with ID: {self.vector_store.id}")
            print(f"File batch status: {file_batch.status}")
            print(f"File counts: {file_batch.file_counts}")
            
            return self.vector_store.id
        
        except Exception as e:
            print(f"Error creating vector store or uploading files: {e}")
            raise
            
        finally:
            # Close all file streams
            [stream.close() for stream in file_stream]



    def create_assistant_with_file_search(self, model: str = "gpt-4o"):
        """
        Create an assistant with file_search tool enabled
        """
        if not self.vector_store:
            raise ValueError("Vector store not created. Call create_vector_store_with_files first.")
        
        if self._is_assistant_exist():
            print("Assistant already exists. Skipping creation.")
            return self.assistant.id
        
        # Create assistant with file_search tool
        self.assistant = self.client.beta.assistants.create(
            name="File Search Assistant",
            instructions="""
            You are an AI assistant of a Rocket launch campaign that qualifies whether or not a person or candidate is suitable for a Rocket launch based on provided documents.
            You have access to a vector store of documents of selection criteria.\n
            Look specially into the occupation and age of data provided in the documents to qualify the candidate. If age is equals or below the age states in documents, it is OK to accept as candidate. \n
            When you are asked a question, search the vector store for relevant documents and use them to answer the question.
            If the provided documents do not contain sufficient information to answer the question, this means candidate is not qualified.\n
            If person is not qualified, response with friendly and non-judgmental tone indicating that the candidate is not qualified for the Rocket launch based on the criteria in the documents.
            \n
            However, if the applicant is very old and is his or her last wish, then approve the applicant regardless of the criteria in the documents.\n
            
            Always provide concise and clear answers.""",
            model=model,
            tools=[{"type": "file_search"}, {"type": "code_interpreter"}],
            tool_resources={
                "file_search": {
                    "vector_store_ids": [self.vector_store.id]
                }
            }
        )
        
        print(f"Assistant created with ID: {self.assistant.id}")
        return self.assistant.id    
    

    def run_assistant(self):
        """
        Run the assistant and wait for completion
        """
        if not self.assistant or not self.thread:
            raise ValueError("Assistant or thread not created.")
        
        # Create a run
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id
        )
        
        # Wait for completion
        while run.status in ["queued", "in_progress"]:
            time.sleep(1)
            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=run.id
            )
            print(f"Run status: {run.status}")
        
        if run.status == "completed":
            # Get messages
            messages = self.client.beta.threads.messages.list(
                thread_id=self.thread.id
            )
            
            # Get the latest assistant message
            for message in messages.data:
                if message.role == "assistant":
                    return message.content[0].text.value
        
        elif run.status == "failed":
            print(f"Run failed: {run.last_error}")
            return None
        
        return None  


    def chat(self, prompt: str) -> str:
        """
        Send a user message to the assistant and get a response
        """

        try:
            if not self.assistant and not self.thread:
                raise ValueError("Assistant not created. Call create_assistant_with_file_search first.")
            
            # self.thread = self.client.beta.threads.create()
            # print(f"Thread created with ID: {self.thread.id}")

            # Add user message to thread
            __message = self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=prompt
            )

            # Run the assistant
            response = self.run_assistant()

            #record in cosmos mongo
            #self.cosmos_mongo.insert_passenger_qualifying_info(prompt, response)

            return response
        
        except Exception as e:
            print(f"Error during chat: {e}")
            raise
        
        finally:
            pass
            # if self.thread:
            #     self.client.beta.threads.delete(self.thread.id)


    def cleanup(self):
        """
        Clean up resources: delete assistant and vector store
        """
        try:
            if self.thread:
                self.client.beta.threads.delete(self.thread.id)
                print(f"Thread {self.thread.id} deleted")

            # if self.assistant:
            #     self.client.beta.assistants.delete(self.assistant.id)
            #     print(f"Assistant {self.assistant.id} deleted")
        
            # if self.vector_store:
            #     self.client.vector_stores.delete(self.vector_store.id)
            #     print(f"Vector store {self.vector_store.id} deleted")
        except Exception as e:
            print(f"Error cleaning up resources: {e}")