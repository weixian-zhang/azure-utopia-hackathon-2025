from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
import uvicorn
from stage_2.vector_db import VectorStore
from stage_2.rag import RAG
from stage_2.evil_llm import EvilLLM
from langchain_openai import AzureChatOpenAI
from stage_3.openai_assistant import OpenAIAssistant
from stage_4.llm_image_ocr import LLMImageOCR
from stage_5.settler_health_agent import SettlerHealthAgent

# {
#   "query": "What is the average heart rate for passenger 18?",
#   "username": "luna_pilot",
#   "team": "team@example.com",
#   "call_index": 4
# }

class RequestData(BaseModel):
    username: str = None
    team: str = None
    call_index: int = None

class Stage5RequestData(RequestData):
    query: str

class ImageBase64Request(BaseModel):
    image_base64: str
    image_format: str = "png"  # jpeg, png, etc.
    passenger_id: str = None

class OCRResponse(BaseModel):
    registered: bool = False
    security_alert: bool = False
    passenger_id: str = ''

class AppState():
    rag: RAG = None
    evil_llm: EvilLLM = None
    llm: AzureChatOpenAI = None
    openai_assistant: OpenAIAssistant = None
    llm_image_ocr: LLMImageOCR = None
    settler_health_agent: SettlerHealthAgent = None

 
app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state.rag = RAG()
    app_state.rag.vectorize_data_if_not_exists()
    app_state.evil_llm = EvilLLM()
    app_state.llm = AzureChatOpenAI(
            deployment_name="gpt-4o",
            model="gpt-4o",
            api_version="2024-12-01-preview",
            temperature=1.0
        )

    app_state.openai_assistant = OpenAIAssistant()
    app_state.openai_assistant.setup_assistant()

    app_state.llm_image_ocr = LLMImageOCR()
    app_state.settler_health_agent = SettlerHealthAgent()

    yield

    # Shutdown: runs when application is shutting down
    print("Shutting down...")
    app_state.rag = None
    app_state.evil_llm = None
    app_state.llm = None
    app_state.openai_assistant = None
    app_state.llm_image_ocr  = None
    app_state.slm_bert = None


app = FastAPI(lifespan=lifespan)


@app.post("/stage-1")
async def stage_1(data: RequestData):
    return app_state.llm.invoke(data.input)


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

    try:
        response: str = app_state.evil_llm.ask_evil_question(data.input)

        return {
            "status": "success",
            "data": response
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/stage-3")
async def rag_with_ai_search(data: RequestData):
    """
    retrieval augmented generation solution using Azure AI Search and Azure OpenAI Service.
    """

    try:
        response: str = app_state.openai_assistant.chat(data.input)

        return {
            "status": "success",
            "data": response
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
    

@app.post("/stage-4", response_model=OCRResponse)
async def ocr_badge(request: ImageBase64Request):
    """
    Extract applicant ID from base64 encoded badge image
    """

    assert request.image_base64, "image_base64 is required"

    try:
        passenger_id = app_state.llm_image_ocr.extract_passenger_badge_id_from_base64_image(
            request.image_base64,
            request.image_format
        )

        return OCRResponse(
            registered=(passenger_id == request.passenger_id) if request.passenger_id else False,
            security_alert=False,
            passenger_id=passenger_id
        )

    
    except Exception as e:
        return {"error": str(e)}
    
    

@app.post("/stage-5")
async def settler_health_agent_invoke(data: Stage5RequestData):
    """
    using Small Language Model for sentiment analysis
    """

    try:
        response: str = app_state.settler_health_agent.invoke(data.query)

        return {
            "response": response
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
    

@app.post("/stage-5-slm")
async def slm_sentiment_analysis(data: RequestData):
    """
    using Small Language Model for sentiment analysis
    """

    try:
        bert_response: str = app_state.slm.distill_bert_analyze_sentiment(data.input)

        phi4mini_response: str = app_state.slm.phi_4_mini_instruct_analyze_sentiment(data.input)

        return {
            "status": "success",
            "data": {
                "distill_bert": bert_response,
                "phi_4_mini": phi4mini_response
            }
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
    

@app.post("/stage-5")
async def sqlite_query_generator(data: RequestData):
    pass
    

@app.get('/')
async def health_check():
    return {"status": "healthy"}

@app.get('/health')
async def health_check():
    return {"status": "healthy"}



# @app.post("/stage-2-evil-llm")
# async def chat_endpoint(data: RequestData):

#     response: str = app_state.evil_llm.ask_evil_question(data.input)
    
#     return {
#         "status": "success",
#         "data": response
#     }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)