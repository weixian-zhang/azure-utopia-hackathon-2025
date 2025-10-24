from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from evil_llm import EvilLLM

class EvilLLMSingleton:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = EvilLLM()
        return cls._instance
 
app = FastAPI()


class RequestData(BaseModel):
    input: str


@app.post("/chat")
async def chat_endpoint(data: RequestData):
    evil_llm = EvilLLMSingleton.get_instance()
    response: str = evil_llm.ask_evil_question(data.input)
    
    return {
        "status": "success",
        "data": response
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)