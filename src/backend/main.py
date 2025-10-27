from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
import uvicorn
from rag import RAG
from evil_llm import EvilLLM
from virtue_llm import VirtueLLM
from openai_assistant import OpenAIAssistant
from llm_image_ocr import LLMImageOCR
from slm import SLM

class RequestData(BaseModel):
    input: str

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
    virtue_llm: VirtueLLM = None
    openai_assistant: OpenAIAssistant = None
    llm_image_ocr: LLMImageOCR = None
    slm: SLM = None

 
app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state.rag = RAG()
    app_state.rag.vectorize_data_if_not_exists()
    app_state.evil_llm = EvilLLM()
    app_state.virtue_llm = VirtueLLM()

    app_state.openai_assistant = OpenAIAssistant()
    app_state.openai_assistant.setup_assistant()

    app_state.llm_image_ocr = LLMImageOCR()
    app_state.slm = SLM()

    yield

    # Shutdown: runs when application is shutting down
    print("Shutting down...")
    app_state.rag = None
    app_state.evil_llm = None
    app_state.virtue_llm = None
    app_state.openai_assistant = None
    app_state.llm_image_ocr  = None
    app_state.slm_bert = None


app = FastAPI(lifespan=lifespan)


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
async def chat_endpoint(data: RequestData):
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
async def chat_endpoint(data: RequestData):
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