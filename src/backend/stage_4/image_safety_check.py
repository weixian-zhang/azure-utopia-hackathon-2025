import requests
import base64
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

class SafetyResponse(BaseModel):
    is_safe: bool = Field(description="Indicates if the image is safe")


class ImageSafetyChecker:

    def __init__(self):
        self.llm = AzureChatOpenAI(
            deployment_name="gpt-4o",
            model="gpt-4o",
            api_version="2024-12-01-preview",
            temperature=0.0
        ).with_structured_output(SafetyResponse)
    

    def download_image_to_base64(self, image_url: str) -> bool:

        response = requests.get(image_url)
        response.raise_for_status()  # Raise error for bad status codes
        
        base64_string = base64.b64encode(response.content).decode('utf-8')

        return base64_string
    

    def invoke(self, image_url: str) -> SafetyResponse:
        base64_image = self.download_image_to_base64(image_url)
        
        prompt = f"Check if the following image (in base64) is safe for for security check: {base64_image}. Respond with a boolean field 'is_safe'."
        
        data_uri = f"data:image/png;base64,{base64_image}"
        
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": """Check if the following image (in base64) contains any harmful objects like scissors, knife, guns or any weapons, or other dangerous items.
                    
                    Any item that cannot typically be carried onto an airplane should be considered harmful.
                    """
                    
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri}
                }
            ]
        )

        response = self.llm.invoke([message])

        return response.is_safe
    

if __name__ == "__main__":
    image_url = "https://azureutopiacontent.blob.core.windows.net/assets/4.png"
    checker = ImageSafetyChecker()
    is_safe = checker.invoke(image_url)
    print(f"Is the image safe? {is_safe}")