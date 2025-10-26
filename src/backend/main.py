from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
import uvicorn
from rag import RAG
from evil_llm import EvilLLM
from virtue_llm import VirtueLLM

# class EvilLLMSingleton:
#     _instance = None
    
#     @classmethod
#     def get_instance(cls):
#         if cls._instance is None:
#             cls._instance = EvilLLM()
#         return cls._instance

class AppState():
    rag: RAG = None
    evil_llm: EvilLLM = None
    virtue_llm: VirtueLLM = None

 
app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state.rag = RAG()
    app_state.rag.vectorize_data_if_not_exists()
    app_state.evil_llm = EvilLLM()
    app_state.virtue_llm = VirtueLLM()

    yield

    # Shutdown: runs when application is shutting down
    print("Shutting down...")
    app_state.rag = None
    app_state.evil_llm = None
    app_state.virtue_llm = None


app = FastAPI(lifespan=lifespan)


class RequestData(BaseModel):
    input: str

@app.post("/stage-1")
async def stage_1(data: RequestData):
    return app_state.virtue_llm.invoke(data.input)

@app.post("/stage-2")
async def chat_endpoint(data: RequestData):
    """
    retrieval augmented generation solution using Azure AI Search and Azure OpenAI Service.
    """
    response: str = app_state.rag.answer_query(data.input)
    
    return {
        "status": "success",
        "data": response
    }


@app.post("/stage-21")
async def chat_endpoint(data: RequestData):
    """
    retrieval augmented generation solution using Azure AI Search and Azure OpenAI Service.
    """
    response: str = app_state.evil_llm.ask_evil_question(data.input)
    
    return {
        "status": "success",
        "data": response
    }


# @app.post("/stage-2-evil-llm")
# async def chat_endpoint(data: RequestData):

#     response: str = app_state.evil_llm.ask_evil_question(data.input)
    
#     return {
#         "status": "success",
#         "data": response
#     }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)